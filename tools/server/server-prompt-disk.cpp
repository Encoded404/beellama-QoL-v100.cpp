#include "server-prompt-disk.h"

#include "log.h"

#include "../../src/llama-mmap.h" // staging: read-only mapping of a published swap file

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>

// the mapping capability probe below reads the same feature macro as
// src/llama-mmap.cpp, which requires these headers
#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
        #endif
    #endif
#endif

#define PCD_TRC(fmt, ...) LOG_TRC("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define PCD_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define PCD_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

namespace server_prompt_disk {

//
// hashing
//

uint64_t hash_bytes(uint64_t seed, const void * data, size_t size) {
    // FNV-1a 64
    constexpr uint64_t offset = 1469598103934665603ull;
    constexpr uint64_t prime  = 1099511628211ull;

    uint64_t h = seed == 0 ? offset : seed;
    const uint8_t * p = (const uint8_t *) data;

    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= prime;
    }

    return h;
}

uint64_t hash_bytes(const void * data, size_t size) {
    return hash_bytes(0, data, size);
}

//
// checkpoint appendix
//

namespace {

struct reader {
    const uint8_t * data;
    size_t          size;
    size_t          pos = 0;

    size_t remaining() const {
        return size - pos;
    }

    bool read_raw(void * dst, size_t n) {
        if (n > remaining()) {
            return false;
        }
        if (n > 0) {
            memcpy(dst, data + pos, n);
        }
        pos += n;
        return true;
    }

    template<typename T>
    bool read(T & v) {
        return read_raw(&v, sizeof(v));
    }
};

template<typename Buffer>
bool read_buffer(reader & r, Buffer & buf) {
    uint64_t n = 0;
    if (!r.read(n) || n > r.remaining()) {
        return false;
    }
    buf.resize(size_t(n));
    return n == 0 || r.read_raw(buf.data(), size_t(n));
}

void append_raw(std::vector<uint8_t> & out, const void * src, size_t n) {
    if (n == 0) {
        return;
    }
    const uint8_t * p = (const uint8_t *) src;
    out.insert(out.end(), p, p + n);
}

template<typename T>
void append(std::vector<uint8_t> & out, const T & v) {
    append_raw(out, &v, sizeof(v));
}

template<typename Buffer>
void append_buffer(std::vector<uint8_t> & out, const Buffer & buf) {
    const uint64_t n = buf.size();
    append(out, n);
    append_raw(out, buf.data(), size_t(n));
}

} // namespace

void write_checkpoints(std::vector<uint8_t> & out, const std::list<common_prompt_checkpoint> & checkpoints) {
    if (checkpoints.empty()) {
        return;
    }

    append(out, CKPT_MAGIC);
    append(out, CKPT_VERSION);
    append(out, (uint32_t) checkpoints.size());

    for (const auto & cur : checkpoints) {
        append(out, cur.n_tokens);
        append(out, cur.pos_min);
        append(out, cur.pos_max);
        append_buffer(out, cur.data_tgt);
        append_buffer(out, cur.data_dft);
        append_buffer(out, cur.data_spec);
    }
}

bool read_checkpoints(const uint8_t * data, size_t size, std::list<common_prompt_checkpoint> & out,
        size_t limit, size_t * n_consumed) {
    if (n_consumed != nullptr) {
        *n_consumed = 0;
    }
    if (size == 0) {
        return true; // valid empty appendix
    }
    if (data == nullptr) {
        return false;
    }

    reader r { data, size, 0 };

    uint32_t magic   = 0;
    uint32_t version = 0;
    uint32_t count   = 0;
    if (!r.read(magic) || magic != CKPT_MAGIC) {
        return false;
    }
    if (!r.read(version) || version != CKPT_VERSION) {
        return false;
    }
    if (!r.read(count)) {
        return false;
    }

    // each entry needs its metadata and three blob lengths at minimum
    constexpr size_t entry_min = sizeof(int64_t) + 2*sizeof(llama_pos) + 3*sizeof(uint64_t);
    if (count > r.remaining()/entry_min) {
        return false;
    }

    std::list<common_prompt_checkpoint> parsed;
    for (uint32_t i = 0; i < count; ++i) {
        common_prompt_checkpoint cur;
        cur.id_task = -1;
        if (!r.read(cur.n_tokens) || !r.read(cur.pos_min) || !r.read(cur.pos_max)) {
            return false;
        }
        if (!read_buffer(r, cur.data_tgt) || !read_buffer(r, cur.data_dft) || !read_buffer(r, cur.data_spec)) {
            return false;
        }
        parsed.push_back(std::move(cur));

        if (limit != SIZE_MAX) {
            while (parsed.size() > limit) {
                parsed.pop_front();
            }
        }
    }

    out = std::move(parsed);
    if (n_consumed != nullptr) {
        *n_consumed = r.pos;
    }
    return true;
}

bool checkpoints_match_prompt(const std::list<common_prompt_checkpoint> & checkpoints,
        size_t expected_count, int64_t prompt_n_tokens, llama_pos prompt_pos_next) {
    if (checkpoints.size() != expected_count) {
        return false;
    }

    for (const auto & cur : checkpoints) {
        if (cur.n_tokens <= 0 || cur.n_tokens > prompt_n_tokens) {
            return false;
        }
        if (cur.pos_min < 0 || cur.pos_max < cur.pos_min) {
            return false;
        }
        if (cur.pos_max >= prompt_pos_next) {
            return false;
        }
        if (cur.empty()) {
            return false;
        }
    }

    return true;
}

//
// file header
//

std::vector<uint8_t> store::encode_header(const header & h) {
    std::vector<uint8_t> out(HEADER_SIZE, 0);

    const auto put32 = [&](size_t off, uint32_t v) { memcpy(out.data() + off, &v, sizeof(v)); };
    const auto put64 = [&](size_t off, uint64_t v) { memcpy(out.data() + off, &v, sizeof(v)); };

    put32(0,  FILE_MAGIC);
    put32(4,  FILE_VERSION);
    put64(8,  h.identity);
    put64(16, h.n_tokens);
    put64(24, h.tokens_hash);
    put64(32, h.off_main);
    put64(40, h.size_main);
    put64(48, h.off_drft);
    put64(56, h.size_drft);
    put64(64, h.off_spec);
    put64(72, h.size_spec);
    put64(80, h.off_ckpt);
    put64(88, h.size_ckpt);
    put64(96, h.payload_hash);

    return out;
}

bool store::decode_header(const uint8_t * data, size_t size, header & h) {
    if (data == nullptr || size < HEADER_SIZE) {
        return false;
    }

    uint32_t magic   = 0;
    uint32_t version = 0;
    memcpy(&magic,   data + 0, sizeof(magic));
    memcpy(&version, data + 4, sizeof(version));

    if (magic != FILE_MAGIC || version != FILE_VERSION) {
        return false;
    }

    const auto get64 = [&](size_t off) {
        uint64_t v = 0;
        memcpy(&v, data + off, sizeof(v));
        return v;
    };

    h.identity     = get64(8);
    h.n_tokens     = get64(16);
    h.tokens_hash  = get64(24);
    h.off_main     = get64(32);
    h.size_main    = get64(40);
    h.off_drft     = get64(48);
    h.size_drft    = get64(56);
    h.off_spec     = get64(64);
    h.size_spec    = get64(72);
    h.off_ckpt     = get64(80);
    h.size_ckpt    = get64(88);
    h.payload_hash = get64(96);

    return true;
}

//
// store
//

mapping::mapping() = default;
mapping::~mapping() = default;
mapping::mapping(mapping &&) noexcept = default;
mapping & mapping::operator=(mapping &&) noexcept = default;

bool store::mapping_supported() {
#if defined(_POSIX_MAPPED_FILES) || defined(_WIN32)
    return true;
#else
    return false;
#endif
}

store::store(config cfg, uint64_t identity) : cfg_(std::move(cfg)), identity_(identity) {
    if (!cfg_.enabled()) {
        return;
    }

    std::error_code ec;
    if (!std::filesystem::is_directory(std::filesystem::u8path(cfg_.path), ec)) {
        PCD_WRN("disk swap directory is not usable, disabling the tier: %s\n", cfg_.path.c_str());
        cfg_.path.clear();
        return;
    }

    if (cfg_.persist) {
        // persistent mode: a previous run's published files are adopted by the
        // caller (store::adopt), so nothing is removed here
    } else {
        wipe();
    }

    if (cfg_.async_writes) {
        worker_ = std::thread([this] { writer_loop(); });
    }
}

store::~store() {
    // let every published spill land before the directory is wiped
    flush();

    if (worker_.joinable()) {
        {
            std::unique_lock lock(mutex_);
            stop_ = true;
        }
        cv_work_.notify_all();
        worker_.join();
    }

    wipe();
}

void store::remove_file(const std::string & path) {
    if (path.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::remove(std::filesystem::u8path(path), ec);
}

void store::wipe() {
    if (cfg_.path.empty()) {
        return;
    }

    if (cfg_.persist) {
        // persistent mode: the directory outlives the process
        return;
    }

    std::error_code ec;
    std::filesystem::directory_iterator it(std::filesystem::u8path(cfg_.path), ec);
    if (ec) {
        return;
    }

    size_t n_removed = 0;
    for (const auto & de : it) {
        const std::string name = de.path().filename().u8string();
        if (name.size() < 8 || name.compare(0, 4, "pcd-") != 0) {
            continue;
        }

        const bool is_swap = name.size() > 5 && name.compare(name.size() - 5, 5, ".swap") == 0;
        const bool is_tmp  = name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp")  == 0;
        if (!is_swap && !is_tmp) {
            continue;
        }

        std::error_code ec_rm;
        if (std::filesystem::remove(de.path(), ec_rm) && !ec_rm) {
            ++n_removed;
        }
    }

    if (n_removed > 0) {
        PCD_INF("removed %zu stale disk swap file(s) from '%s'\n", n_removed, cfg_.path.c_str());
    }

    {
        std::unique_lock lock(mutex_);
        cancelled_.clear();
        in_flight_.clear();
    }

    bytes_on_disk_   = 0;
    entries_on_disk_ = 0;
}

void store::flush() {
    if (!worker_.joinable()) {
        return;
    }

    std::unique_lock lock(mutex_);
    cv_done_.wait(lock, [this] {
        return queue_.empty() && !busy_;
    });
}

bool store::write_job(const job & j) {
    {
        std::ofstream ofs(std::filesystem::u8path(j.tmp_path), std::ios::binary | std::ios::trunc);
        if (!ofs) {
            PCD_WRN("failed to open disk swap file for writing: %s\n", j.tmp_path.c_str());
            return false;
        }

        const auto put = [&](const void * p, size_t n) {
            if (n > 0) {
                ofs.write((const char *) p, n);
            }
        };

        put(j.header.data(), j.header.size());
        put(j.main.data(),   j.main.size());
        put(j.drft.data(),   j.drft.size());
        put(j.spec.data(),   j.spec.size());
        put(j.ckpt.data(),   j.ckpt.size());

        ofs.flush();
        ofs.close();

        if (!ofs) {
            PCD_WRN("failed to write disk swap file: %s\n", j.tmp_path.c_str());
            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(std::filesystem::u8path(j.tmp_path), std::filesystem::u8path(j.final_path), ec);
    if (ec) {
        PCD_WRN("failed to publish disk swap file: %s\n", j.final_path.c_str());
        return false;
    }

    return true;
}

void store::writer_loop() {
    for (;;) {
        job j;
        {
            std::unique_lock lock(mutex_);
            cv_work_.wait(lock, [this] {
                return stop_ || !queue_.empty();
            });
            if (queue_.empty()) {
                if (stop_) {
                    return;
                }
                continue;
            }
            j = std::move(queue_.front());
            queue_.pop_front();
            busy_ = true;
        }

        const bool ok = write_job(j);

        {
            std::unique_lock lock(mutex_);
            busy_ = false;

            in_flight_.erase(j.final_path);

            if (!ok) {
                remove_file(j.tmp_path);
                ++spill_failures;
            } else if (cancelled_.erase(j.final_path) > 0) {
                // the entry was erased while this write was still in flight
                remove_file(j.final_path);
            } else {
                ++spill_successes;
                PCD_TRC("spilled %.3f MiB to '%s'\n", j.total/1024.0/1024.0, j.final_path.c_str());
            }
        }

        cv_done_.notify_all();
    }
}

bool store::submit(job && j) {
    if (!worker_.joinable()) {
        return write_job(j);
    }

    {
        std::unique_lock lock(mutex_);
        in_flight_.insert(j.final_path);
        queue_.push_back(std::move(j));
    }
    cv_work_.notify_one();
    return true;
}

void store::erase(entry & e) {
    if (e.on_disk()) {
        remove_file(e.path);

        if (worker_.joinable()) {
            std::unique_lock lock(mutex_);
            // the file may not exist yet; the writer removes it once it lands
            if (in_flight_.count(e.path) > 0) {
                cancelled_.insert(e.path);
            }
        }

        const size_t bytes_cur = bytes_on_disk_.load();
        bytes_on_disk_ = bytes_cur > e.bytes ? bytes_cur - e.bytes : 0;

        const size_t entries_cur = entries_on_disk_.load();
        entries_on_disk_ = entries_cur > 0 ? entries_cur - 1 : 0;
    }

    e = entry {};
}

bool store::spill(entry & e,
        const uint8_t * main, size_t size_main,
        const uint8_t * drft, size_t size_drft,
        const uint8_t * spec, size_t size_spec,
        const std::list<common_prompt_checkpoint> & checkpoints,
        uint64_t n_tokens, uint64_t tokens_hash) {
    std::vector<uint8_t> v_main;
    std::vector<uint8_t> v_drft;
    std::vector<uint8_t> v_spec;

    try {
        if (size_main > 0) {
            v_main.assign(main, main + size_main);
        }
        if (size_drft > 0) {
            v_drft.assign(drft, drft + size_drft);
        }
        if (size_spec > 0) {
            v_spec.assign(spec, spec + size_spec);
        }
    } catch (const std::bad_alloc &) {
        ++spill_failures;
        return false;
    }

    return spill(e, std::move(v_main), std::move(v_drft), std::move(v_spec), checkpoints, n_tokens, tokens_hash);
}

bool store::spill(entry & e,
        std::vector<uint8_t> && main,
        std::vector<uint8_t> && drft,
        std::vector<uint8_t> && spec,
        const std::list<common_prompt_checkpoint> & checkpoints,
        uint64_t n_tokens, uint64_t tokens_hash) {
    if (!enabled()) {
        return false;
    }

    ++spill_attempts;

    if (e.on_disk() || main.empty()) {
        ++spill_failures;
        return false;
    }

    job j;

    try {
        write_checkpoints(j.ckpt, checkpoints);
    } catch (const std::bad_alloc &) {
        ++spill_failures;
        return false;
    }

    j.main = std::move(main);
    j.drft = std::move(drft);
    j.spec = std::move(spec);

    const uint64_t total = HEADER_SIZE + j.main.size() + j.drft.size() + j.spec.size() + j.ckpt.size();
    if (total > limit_bytes()) {
        ++spill_failures;
        PCD_WRN("disk swap entry (%.3f MiB) exceeds the disk budget (%.3f MiB), not spilling\n",
                total/1024.0/1024.0, limit_bytes()/1024.0/1024.0);
        return false;
    }

    header h;
    h.identity     = identity_;
    h.n_tokens     = n_tokens;
    h.tokens_hash  = tokens_hash;
    h.off_main     = HEADER_SIZE;
    h.size_main    = j.main.size();
    h.off_drft     = h.off_main + h.size_main;
    h.size_drft    = j.drft.size();
    h.off_spec     = h.off_drft + h.size_drft;
    h.size_spec    = j.spec.size();
    h.off_ckpt     = h.off_spec + h.size_spec;
    h.size_ckpt    = j.ckpt.size();

    uint64_t payload_hash = hash_bytes(0, j.main.data(), j.main.size());
    payload_hash = hash_bytes(payload_hash, j.drft.data(), j.drft.size());
    payload_hash = hash_bytes(payload_hash, j.spec.data(), j.spec.size());
    payload_hash = hash_bytes(payload_hash, j.ckpt.data(), j.ckpt.size());
    h.payload_hash = payload_hash;

    const char * file_prefix = cfg_.persist ? "pcp-" : "pcd-";
    const std::string final_path = cfg_.path + file_prefix + std::to_string(next_serial_++) + ".swap";
    const std::string tmp_path   = final_path + ".tmp";

    j.header     = encode_header(h);
    j.total      = total;
    j.final_path = final_path;
    j.tmp_path   = tmp_path;

    if (!submit(std::move(j))) {
        remove_file(tmp_path);
        ++spill_failures;
        PCD_WRN("failed to write disk swap file: %s\n", final_path.c_str());
        return false;
    }

    if (!worker_.joinable()) {
        // synchronous write already landed
        ++spill_successes;
    }

    // Publish the entry. Under async writes the bytes are owned by the writer;
    // a read that arrives first drains the writer instead of failing.
    e.status      = entry::status::present;
    e.path        = final_path;
    e.bytes       = total;
    e.size_main   = h.size_main;
    e.size_drft   = h.size_drft;
    e.size_spec   = h.size_spec;
    e.size_ckpt   = h.size_ckpt;
    e.n_tokens    = n_tokens;
    e.tokens_hash = tokens_hash;

    bytes_on_disk_   += size_t(total);
    entries_on_disk_ += 1;

    return true;
}

std::unique_ptr<mapping> store::open(entry & e) {
    if (!enabled() || !e.on_disk() || e.path.empty()) {
        return nullptr;
    }

    ++restore_attempts;

    const auto fail = [&](const char * reason) -> std::unique_ptr<mapping> {
        PCD_WRN("rejecting disk swap file (%s), entry invalidated: %s\n", reason, e.path.c_str());
        erase(e);
        ++restore_failures;
        return nullptr;
    };

    // A published entry may not have reached disk yet when writes are deferred.
    // Draining is bounded and only happens on this race.
    {
        std::error_code ec_exists;
        if (!std::filesystem::exists(std::filesystem::u8path(e.path), ec_exists)) {
            flush();
            if (!std::filesystem::exists(std::filesystem::u8path(e.path), ec_exists)) {
                return fail("missing");
            }
        }
    }

    std::ifstream ifs(std::filesystem::u8path(e.path), std::ios::binary);
    if (!ifs) {
        return fail("unreadable");
    }

    uint8_t head[HEADER_SIZE];
    if (!ifs.read((char *) head, sizeof(head))) {
        return fail("truncated header");
    }

    header h;
    if (!decode_header(head, sizeof(head), h)) {
        return fail("bad file header");
    }

    if (h.identity != identity_) {
        return fail("identity mismatch");
    }

    if (h.n_tokens != e.n_tokens || h.tokens_hash != e.tokens_hash) {
        return fail("token correlation mismatch");
    }

    if (h.size_main != e.size_main || h.size_drft != e.size_drft ||
            h.size_spec != e.size_spec || h.size_ckpt != e.size_ckpt) {
        return fail("section layout mismatch");
    }

    if (h.off_main != HEADER_SIZE ||
            h.off_drft != h.off_main + h.size_main ||
            h.off_spec != h.off_drft + h.size_drft ||
            h.off_ckpt != h.off_spec + h.size_spec) {
        return fail("non-contiguous sections");
    }

    const uint64_t expected_total = h.off_ckpt + h.size_ckpt;
    if (expected_total != e.bytes || h.size_main == 0) {
        return fail("size mismatch");
    }

    mapping m;

    // Prefer mapping: the payload then lives in reclaimable page cache instead of
    // an anonymous buffer that doubles the restore footprint.
    if (cfg_.use_mmap) {
        try {
            m.file_ = std::make_unique<llama_file>(e.path.c_str(), "rb");
            if (m.file_->size() < expected_total) {
                m.file_.reset();
            } else {
                // prefetch = 0: let the reads below fault the pages in on demand
                m.map_ = std::make_unique<llama_mmap>(m.file_.get(), 0);
                if (m.map_->size() < expected_total) {
                    m.map_.reset();
                    m.file_.reset();
                }
            }
        } catch (const std::exception & err) {
            PCD_TRC("mapping unavailable, using a buffered read: %s\n", err.what());
            m.map_.reset();
            m.file_.reset();
        }
    }

    uint64_t payload_hash = 0;

    if (m.map_) {
        const uint8_t * base = (const uint8_t *) m.map_->addr();

        m.ckpt_owned_.assign(base + h.off_ckpt, base + h.off_ckpt + h.size_ckpt);

        m.main_ = base + h.off_main; m.size_main_ = size_t(h.size_main);
        m.drft_ = base + h.off_drft; m.size_drft_ = size_t(h.size_drft);
        m.spec_ = base + h.off_spec; m.size_spec_ = size_t(h.size_spec);
        m.mapped_ = true;

        payload_hash = hash_bytes(0, m.main_, m.size_main_);
        payload_hash = hash_bytes(payload_hash, m.drft_, m.size_drft_);
        payload_hash = hash_bytes(payload_hash, m.spec_, m.size_spec_);
        payload_hash = hash_bytes(payload_hash, m.ckpt_owned_.data(), m.ckpt_owned_.size());

        if (payload_hash != h.payload_hash) {
            // release the mapping before deleting the file
            m = mapping {};
            return fail("payload hash mismatch");
        }

        PCD_TRC("mapped %.3f MiB from '%s'\n", expected_total/1024.0/1024.0, e.path.c_str());
    } else {
        // buffered fallback: the header was already consumed from the stream
        m.owned_.resize(size_t(h.size_main + h.size_drft + h.size_spec));
        m.ckpt_owned_.resize(size_t(h.size_ckpt));

        if ((!m.owned_.empty() && !ifs.read((char *) m.owned_.data(), m.owned_.size())) ||
                (!m.ckpt_owned_.empty() && !ifs.read((char *) m.ckpt_owned_.data(), m.ckpt_owned_.size()))) {
            return fail("truncated payload");
        }

        if (ifs.peek() != std::char_traits<char>::eof()) {
            return fail("trailing bytes");
        }

        payload_hash = hash_bytes(0, m.owned_.data(), m.owned_.size());
        payload_hash = hash_bytes(payload_hash, m.ckpt_owned_.data(), m.ckpt_owned_.size());

        if (payload_hash != h.payload_hash) {
            return fail("payload hash mismatch");
        }

        m.main_ = m.owned_.data();       m.size_main_ = size_t(h.size_main);
        m.drft_ = m.main_ + h.size_main; m.size_drft_ = size_t(h.size_drft);
        m.spec_ = m.drft_ + h.size_drft; m.size_spec_ = size_t(h.size_spec);
    }

    ++restore_successes;
    return std::make_unique<mapping>(std::move(m));
}

std::string store::next_name() {
    if (!enabled()) {
        return {};
    }

    const char * file_prefix = cfg_.persist ? "pcp-" : "pcd-";
    return cfg_.path + file_prefix + std::to_string(next_serial_++) + ".swap";
}

bool store::adopt(entry & e) {
    if (!enabled() || !e.on_disk() || e.path.empty()) {
        return false;
    }

    std::ifstream ifs(std::filesystem::u8path(e.path), std::ios::binary);
    if (!ifs) {
        return false;
    }

    uint8_t head[HEADER_SIZE];
    if (!ifs.read((char *) head, sizeof(head))) {
        return false;
    }

    header h;
    if (!decode_header(head, sizeof(head), h)) {
        return false;
    }

    if (h.identity != identity_) {
        return false;
    }

    // the caller's entry metadata must correlate with the file, exactly like a
    // read of a runtime entry would validate it
    if (h.n_tokens != e.n_tokens || h.tokens_hash != e.tokens_hash) {
        return false;
    }

    if (h.off_main != HEADER_SIZE ||
            h.off_drft != h.off_main + h.size_main ||
            h.off_spec != h.off_drft + h.size_drft ||
            h.off_ckpt != h.off_spec + h.size_spec) {
        return false;
    }

    if (h.size_main == 0) {
        return false;
    }

    // advance the serial past the file's number, so nothing later overwrites it
    {
        const std::string & path = e.path;
        const size_t base = path.find("pcp-");
        if (base != std::string::npos) {
            size_t idx = base + 4;
            uint64_t value = 0;
            while (idx < path.size() && path[idx] >= '0' && path[idx] <= '9') {
                value = value*10 + uint64_t(path[idx] - '0');
                ++idx;
                if (value > 0xFFFFFFFFFFFFFFFULL) { // absurd serial: refuse
                    value = 0;
                    break;
                }
            }
            if (value > 0 && value >= next_serial_) {
                next_serial_ = value + 1;
            }
        }
    }

    e.bytes       = h.off_ckpt + h.size_ckpt;
    e.size_main   = h.size_main;
    e.size_drft   = h.size_drft;
    e.size_spec   = h.size_spec;
    e.size_ckpt   = h.size_ckpt;

    // adoption is the persistent tier's registration path
    e.persist_tier = true;

    bytes_on_disk_   += size_t(e.bytes);
    entries_on_disk_ += 1;

    return true;
}

void store::release(const entry & e) {
    if (!e.on_disk()) {
        return;
    }

    const size_t bytes_cur = bytes_on_disk_.load();
    bytes_on_disk_ = bytes_cur > e.bytes ? bytes_cur - e.bytes : 0;

    const size_t entries_cur = entries_on_disk_.load();
    entries_on_disk_ = entries_cur > 0 ? entries_cur - 1 : 0;
}

} // namespace server_prompt_disk
