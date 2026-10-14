#include "server-prompt-persist.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <system_error>
#include <type_traits>

namespace server_prompt_persist {

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
        static_assert(std::is_trivially_copyable<T>::value, "read needs a trivially copyable type");
        return read_raw(&v, sizeof(v));
    }
};

struct writer {
    std::vector<uint8_t> data;

    void append_raw(const void * src, size_t n) {
        if (n == 0) {
            return;
        }
        const uint8_t * p = (const uint8_t *) src;
        data.insert(data.end(), p, p + n);
    }

    template<typename T>
    void append(const T & v) {
        static_assert(std::is_trivially_copyable<T>::value, "append needs a trivially copyable type");
        append_raw(&v, sizeof(v));
    }
};

constexpr char INDEX_NAME[]      = "index.bin";
constexpr char INDEX_NAME_TMP[]  = "index.bin.tmp";

bool valid_record_name(const std::string & name) {
    // published entry files inside the persist directory only
    if (name.size() < 10 || name.compare(0, 4, "pcp-") != 0 ||
            name.compare(name.size() - 5, 5, ".swap") != 0) {
        return false;
    }
    for (size_t i = 4; i + 5 < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return false;
        }
    }
    return true;
}

} // namespace

bool index_read(const std::string & dir, std::vector<record> & records) {
    records.clear();

    std::ifstream ifs(std::filesystem::u8path(dir + INDEX_NAME), std::ios::binary);
    if (!ifs) {
        // no index at all: a clean miss, the caller clears the directory
        return false;
    }

    std::vector<uint8_t> data;
    try {
        data.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    } catch (const std::bad_alloc &) {
        return false;
    }
    if (data.size() < 16) {
        return false;
    }

    reader r { data.data(), data.size(), 0 };

    uint32_t magic   = 0;
    uint32_t version = 0;
    uint64_t identity = 0;
    uint32_t count   = 0;
    if (!r.read(magic) || magic != INDEX_MAGIC) {
        return false;
    }
    if (!r.read(version) || version != INDEX_VERSION) {
        return false;
    }
    if (!r.read(identity) || identity == 0) {
        return false;
    }
    if (!r.read(count)) {
        return false;
    }

    // reject before any parsing, so a corrupt count cannot request unbounded work
    if (count > r.remaining() / INDEX_RECORD_MIN) {
        return false;
    }

    std::vector<record> parsed;
    parsed.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        record rec;

        uint16_t name_len = 0;
        if (!r.read(name_len) || name_len > r.remaining()) {
            return false;
        }
        rec.name.resize(name_len);
        if (name_len > 0 && !r.read_raw(rec.name.data(), name_len)) {
            return false;
        }
        if (!valid_record_name(rec.name)) {
            return false;
        }

        uint8_t flags = 0;
        if (!r.read(flags) || (flags & ~1u)) {
            return false;
        }
        rec.has_media = (flags & 1) != 0;

        uint32_t tokens_len = 0;
        if (!r.read(tokens_len) || tokens_len % sizeof(llama_token) != 0 || tokens_len > r.remaining()) {
            return false;
        }
        rec.tokens.resize(tokens_len);
        if (tokens_len > 0 && !r.read_raw(rec.tokens.data(), tokens_len)) {
            return false;
        }

        uint32_t n_ckpts = 0;
        if (!r.read(n_ckpts) || n_ckpts > r.remaining() / sizeof(ckpt_meta)) {
            return false;
        }
        rec.checkpoints.resize(n_ckpts);
        for (auto & ckpt : rec.checkpoints) {
            if (!r.read(ckpt.n_tokens) || !r.read(ckpt.pos_min) || !r.read(ckpt.pos_max)) {
                return false;
            }
        }

        parsed.push_back(std::move(rec));
    }

    records = std::move(parsed);
    return true;
}

bool index_write(const std::string & dir, uint64_t identity, const std::vector<record> & records) {
    writer w;
    try {
        w.append(INDEX_MAGIC);
        w.append(INDEX_VERSION);
        // identity itself is validated per file at adoption; the index copy is
        // belt and braces for diagnostics
        w.append(identity);
        w.append((uint32_t) records.size());

        for (const auto & rec : records) {
            const uint16_t name_len = (uint16_t) rec.name.size();
            w.append(name_len);
            w.append_raw(rec.name.data(), rec.name.size());

            const uint8_t flags = rec.has_media ? 1 : 0;
            w.append(flags);

            const uint32_t tokens_len = (uint32_t) rec.tokens.size();
            w.append(tokens_len);
            w.append_raw(rec.tokens.data(), rec.tokens.size());

            w.append((uint32_t) rec.checkpoints.size());
            for (const auto & ckpt : rec.checkpoints) {
                w.append(ckpt.n_tokens);
                w.append(ckpt.pos_min);
                w.append(ckpt.pos_max);
            }
        }
    } catch (const std::bad_alloc &) {
        return false;
    }

    {
        std::ofstream ofs(std::filesystem::u8path(dir + INDEX_NAME_TMP), std::ios::binary | std::ios::trunc);
        if (!ofs) {
            return false;
        }
        ofs.write((const char *) w.data.data(), w.data.size());
        ofs.flush();
        ofs.close();
        if (!ofs) {
            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(std::filesystem::u8path(dir + INDEX_NAME_TMP),
            std::filesystem::u8path(dir + INDEX_NAME), ec);
    if (ec) {
        std::filesystem::remove(std::filesystem::u8path(dir + INDEX_NAME_TMP), ec);
        return false;
    }

    return true;
}

size_t remove_stale_files(const std::string & dir, const std::vector<record> & keep) {
    std::error_code ec;
    std::filesystem::directory_iterator it(std::filesystem::u8path(dir), ec);
    if (ec) {
        return 0;
    }

    size_t n_removed = 0;
    for (const auto & de : it) {
        const std::string name = de.path().filename().u8string();

        bool referenced = false;
        if (name == INDEX_NAME) {
            referenced = true;
        } else {
            for (const auto & rec : keep) {
                if (rec.name == name) {
                    referenced = true;
                    break;
                }
            }
        }

        const bool ours = (name.compare(0, 4, "pcp-") == 0 && name.size() > 4 &&
                    (name.compare(name.size() - 5, 5, ".swap") == 0 || name.compare(name.size() - 4, 4, ".tmp") == 0));
        if (!ours || referenced) {
            continue;
        }

        std::error_code ec_rm;
        if (std::filesystem::remove(de.path(), ec_rm) && !ec_rm) {
            ++n_removed;
        }
    }

    return n_removed;
}

void remove_file(const std::string & path) {
    if (path.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::remove(std::filesystem::u8path(path), ec);
}

uint64_t file_fingerprint(const std::string & path) {
    if (path.empty()) {
        return 0;
    }

    std::error_code ec;
    const auto status = std::filesystem::status(std::filesystem::u8path(path), ec);
    if (ec || !std::filesystem::is_regular_file(status)) {
        return 0;
    }

    std::ifstream ifs(std::filesystem::u8path(path), std::ios::binary);
    if (!ifs) {
        return 0;
    }

    ifs.seekg(0, std::ios::end);
    const std::streamoff end = ifs.tellg();
    if (end <= 0) {
        return 0;
    }
    const uint64_t size = (uint64_t) end;

    // the last-write time as a raw tick count: implementation-defined, but
    // stable across runs on the same platform, which is all the identity
    // needs. Any re-generated file (a re-finetune with the same name) lands a
    // new mtime, so content changes are detected even when the changed bytes
    // fall between the sampled slices below.
    uint64_t mtime = 0;
    {
        std::error_code ec_mt;
        const auto wt = std::filesystem::last_write_time(std::filesystem::u8path(path), ec_mt);
        if (!ec_mt) {
            mtime = (uint64_t) wt.time_since_epoch().count();
        }
    }

    constexpr size_t slice = 64 * 1024;
    constexpr int n_slices = 16;

    std::vector<char> buf(slice);
    uint64_t hash = server_prompt_disk::hash_bytes(&size, sizeof(size));
    hash = server_prompt_disk::hash_bytes(hash, &mtime, sizeof(mtime));

    for (int i = 0; i <= n_slices; ++i) {
        // fixed, size-derived offsets: 0, evenly spread, and the file tail
        const uint64_t off = i == n_slices
                ? (size > slice ? size - slice : 0)
                : size * (uint64_t) i / (uint64_t) (n_slices + 1);

        ifs.clear();
        ifs.seekg((std::streamoff) off, std::ios::beg);
        const size_t take = std::min<uint64_t>(slice, size - off);
        if (!ifs.read(buf.data(), take)) {
            // a read that fails on a file that already reported its size is
            // itself distinguishing information
            const uint64_t bad = 0xDEADBEEFull;
            return server_prompt_disk::hash_bytes(hash, &bad, sizeof(bad));
        }
        hash = server_prompt_disk::hash_bytes(hash, buf.data(), take);
    }

    return hash;
}

bool move_file(const std::string & src, const std::string & dst) {
    if (src.empty() || dst.empty() || src == dst) {
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(std::filesystem::u8path(src), std::filesystem::u8path(dst), ec);
    if (!ec) {
        return true;
    }

    // different filesystems: a full copy is the only option. The payload is
    // re-written here; this path is the exception, not the design.
    {
        std::ifstream ifs(std::filesystem::u8path(src), std::ios::binary);
        if (!ifs) {
            return false;
        }
        std::ofstream ofs(std::filesystem::u8path(dst), std::ios::binary | std::ios::trunc);
        if (!ofs) {
            return false;
        }
        char buf[64 * 1024];
        while (ifs.read(buf, sizeof(buf)) || ifs.gcount() > 0) {
            ofs.write(buf, ifs.gcount());
            if (!ofs) {
                return false;
            }
            if (!ifs) {
                break;
            }
        }
        ofs.flush();
        ofs.close();
        if (!ofs) {
            return false;
        }
    }

    std::error_code ec_rm;
    return std::filesystem::remove(std::filesystem::u8path(src), ec_rm) && !ec_rm;
}

bool patch_header_identity(const std::string & path, uint64_t identity) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(std::filesystem::u8path(path), ec);
    if (ec || size < server_prompt_disk::HEADER_SIZE) {
        return false;
    }

    {
        std::fstream fs(std::filesystem::u8path(path), std::ios::binary | std::ios::in | std::ios::out);
        if (!fs) {
            return false;
        }

        uint8_t head[server_prompt_disk::HEADER_SIZE];
        if (!fs.read((char *) head, sizeof(head))) {
            return false;
        }

        // structural check only: magic and version must already be right
        uint32_t magic   = 0;
        uint32_t version = 0;
        memcpy(&magic,   head + 0, sizeof(magic));
        memcpy(&version, head + 4, sizeof(version));
        if (magic != server_prompt_disk::FILE_MAGIC || version != server_prompt_disk::FILE_VERSION) {
            return false;
        }
        if (identity == 0) {
            return false;
        }

        fs.seekp(8, std::ios::beg);
        fs.write((const char *) &identity, sizeof(identity));
        fs.flush();
        fs.close();
        if (!fs) {
            return false;
        }
    }

    return true;
}

} // namespace server_prompt_persist