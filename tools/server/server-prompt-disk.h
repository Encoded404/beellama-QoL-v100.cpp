#pragma once

// Automatic disk swap tier for the server prompt cache.
//
// The tier stores the already-serialized, self-contained sequence state that
// the RAM prompt cache produced (target / owned draft / speculative), plus the
// SCKP checkpoint appendix for the same cache entry. It performs no
// serialization of its own and owns no llama_context.
//
// Invariants:
//   - Files are published atomically: a complete temporary file is renamed into
//     place, so a reader never observes a partially written entry.
//   - Every read is validated (magic, version, identity fingerprint, section
//     offsets, payload hash, token correlation). Any mismatch invalidates and
//     deletes the file rather than reinterpreting bytes.
//   - An invalidated entry is left with no payload at all, so callers must drop
//     it from their cache rather than retry it.
//   - The tier is session-scoped: the directory is owned by this process and is
//     wiped at construction and destruction.

#include "common.h"
#include "llama.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

struct llama_file;
struct llama_mmap;

namespace server_prompt_disk {

// SCKP checkpoint appendix, shared with the manual slot save/load path.
constexpr uint32_t CKPT_MAGIC   = 0x504b4353;
constexpr uint32_t CKPT_VERSION = 1;

inline constexpr uint32_t FILE_MAGIC   = 0x50434453; // 'PCDS'
inline constexpr uint32_t FILE_VERSION = 1;
inline constexpr size_t   HEADER_SIZE  = 128;

// FNV-1a 64. Deterministic and platform independent.
uint64_t hash_bytes(uint64_t seed, const void * data, size_t size);
uint64_t hash_bytes(const void * data, size_t size);

// Append the SCKP appendix for `checkpoints` to `out`. Writes nothing when the
// list is empty.
void write_checkpoints(std::vector<uint8_t> & out, const std::list<common_prompt_checkpoint> & checkpoints);

// Parse an SCKP appendix from [data, data + size). Structural validation only:
// the caller must additionally validate positions against the prompt. `limit`
// truncates the oldest entries, matching the manual slot-save behaviour; pass
// SIZE_MAX for no limit. On failure `out` is unchanged. `n_consumed` reports the
// bytes belonging to the appendix (zero-sized input is a valid empty appendix).
bool read_checkpoints(const uint8_t * data, size_t size, std::list<common_prompt_checkpoint> & out,
        size_t limit, size_t * n_consumed);

// Semantic validation of a parsed appendix against the prompt it belongs to.
// An empty appendix is only valid when `expected_count` is zero.
bool checkpoints_match_prompt(const std::list<common_prompt_checkpoint> & checkpoints,
        size_t expected_count, int64_t prompt_n_tokens, llama_pos prompt_pos_next);

struct config {
    // directory with a trailing separator; empty disables the tier
    std::string path;

    // < 0: no limit, 0: disabled, otherwise MiB
    int64_t limit_mib = 0;

    // Map the payload instead of buffering it. Mapping keeps the payload in
    // reclaimable page cache, which is the point of the tier; falls back to a
    // buffered read when mapping is unavailable.
    bool use_mmap = true;

    // Publish spills from a dedicated writer thread. Spills happen at request
    // admission, never during a decode, so this only hides the write latency.
    bool async_writes = true;

    bool enabled() const {
        return !path.empty() && limit_mib != 0;
    }

    size_t limit_bytes() const {
        if (limit_mib < 0) {
            return SIZE_MAX;
        }
        return size_t(limit_mib) * 1024ull * 1024ull;
    }
};

struct entry {
    enum class status {
        none,    // no payload is stored
        present, // payload is fully written and readable
    };

    status status = status::none;

    std::string path;
    uint64_t    bytes = 0;

    // section layout, kept so restore can validate without re-reading the header
    uint64_t size_main = 0;
    uint64_t size_drft = 0;
    uint64_t size_spec = 0;
    uint64_t size_ckpt = 0;

    // correlation with the owning cache entry
    uint64_t n_tokens    = 0;
    uint64_t tokens_hash = 0;

    bool on_disk() const {
        return status == status::present;
    }
};

// RAII view over an opened entry. The pointers remain valid for the lifetime of
// the object, which must outlive any restore transaction reading them. The
// payload is either mapped (page cache) or buffered, depending on availability.
class mapping {
public:
    mapping();
    ~mapping();

    mapping(mapping &&) noexcept;
    mapping & operator=(mapping &&) noexcept;

    mapping(const mapping &) = delete;
    mapping & operator=(const mapping &) = delete;

    bool mapped() const {
        return mapped_;
    }

    const uint8_t * main() const { return main_; }
    size_t main_size() const { return size_main_; }

    const uint8_t * drft() const { return drft_; }
    size_t drft_size() const { return size_drft_; }

    const uint8_t * spec() const { return spec_; }
    size_t spec_size() const { return size_spec_; }

    const uint8_t * ckpt() const { return ckpt_owned_.data(); }
    size_t ckpt_size() const { return ckpt_owned_.size(); }

private:
    friend class store;

    // declared before map_ so the mapping is torn down before the file handle
    std::unique_ptr<llama_file> file_;
    std::unique_ptr<llama_mmap> map_;

    std::vector<uint8_t> owned_;      // buffered payload: main + drft + spec
    std::vector<uint8_t> ckpt_owned_; // checkpoint appendix, always read into memory

    bool mapped_ = false;

    const uint8_t * main_ = nullptr; size_t size_main_ = 0;
    const uint8_t * drft_ = nullptr; size_t size_drft_ = 0;
    const uint8_t * spec_ = nullptr; size_t size_spec_ = 0;
};

class store {
public:
    store(config cfg, uint64_t identity);
    ~store();

    // Whether payload mapping is available on this platform. When it is not,
    // open() falls back to a buffered read.
    static bool mapping_supported();

    bool enabled() const {
        return cfg_.enabled();
    }

    uint64_t identity() const {
        return identity_;
    }

    const std::string & path() const {
        return cfg_.path;
    }

    size_t limit_bytes() const {
        return cfg_.limit_bytes();
    }

    // Serialize and publish `entry`. Returns false and leaves `entry` unchanged
    // when the tier is disabled or the entry cannot fit the budget.
    //
    // The pointer form copies the payload; the rvalue form takes ownership, which
    // is what demotion uses so a spill costs no extra copy. On success the entry
    // is published and the payload is owned by the writer (or already on disk),
    // so a caller must not expect to keep its buffers.
    bool spill(entry & e,
            const uint8_t * main, size_t size_main,
            const uint8_t * drft, size_t size_drft,
            const uint8_t * spec, size_t size_spec,
            const std::list<common_prompt_checkpoint> & checkpoints,
            uint64_t n_tokens, uint64_t tokens_hash);

    bool spill(entry & e,
            std::vector<uint8_t> && main,
            std::vector<uint8_t> && drft,
            std::vector<uint8_t> && spec,
            const std::list<common_prompt_checkpoint> & checkpoints,
            uint64_t n_tokens, uint64_t tokens_hash);

    // Open a present entry. On any failure the entry is invalidated (no payload
    // remains) and its file is deleted; returns null.
    std::unique_ptr<mapping> open(entry & e);

    // Invalidate an entry and delete its file. Safe on any status.
    void erase(entry & e);

    // Wait for every published spill to reach disk. A read of a not-yet-written
    // entry drains automatically, so callers only need this for shutdown or to
    // observe files deterministically.
    void flush();

    size_t bytes_on_disk() const {
        return bytes_on_disk_.load();
    }

    size_t entries_on_disk() const {
        return entries_on_disk_.load();
    }

    std::atomic<uint64_t> spill_attempts{0};
    std::atomic<uint64_t> spill_successes{0};
    std::atomic<uint64_t> spill_failures{0};
    std::atomic<uint64_t> restore_attempts{0};
    std::atomic<uint64_t> restore_successes{0};
    std::atomic<uint64_t> restore_failures{0};

private:
    struct header {
        uint64_t identity    = 0;
        uint64_t n_tokens    = 0;
        uint64_t tokens_hash = 0;
        uint64_t off_main    = 0;
        uint64_t size_main   = 0;
        uint64_t off_drft    = 0;
        uint64_t size_drft   = 0;
        uint64_t off_spec    = 0;
        uint64_t size_spec   = 0;
        uint64_t off_ckpt    = 0;
        uint64_t size_ckpt   = 0;
        uint64_t payload_hash = 0;
    };

    struct job {
        std::string tmp_path;
        std::string final_path;

        std::vector<uint8_t> header;
        std::vector<uint8_t> main;
        std::vector<uint8_t> drft;
        std::vector<uint8_t> spec;
        std::vector<uint8_t> ckpt;

        uint64_t total = 0;
    };

    static std::vector<uint8_t> encode_header(const header & h);
    static bool                 decode_header(const uint8_t * data, size_t size, header & h);

    // delete a file, ignoring absence
    static void remove_file(const std::string & path);

    // remove every *.swap / *.tmp this tier owns
    void wipe();

    // write one job's bytes and publish it; false on any I/O error
    bool write_job(const job & j);

    void writer_loop();

    // publish a prepared job, synchronously or through the writer
    bool submit(job && j);

    config   cfg_;
    uint64_t identity_ = 0;

    uint64_t next_serial_ = 0;

    std::mutex              mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::deque<job>         queue_;
    bool                    stop_ = false;
    bool                    busy_ = false;
    std::thread             worker_;

    // paths with a queued or running write, and the subset of those whose entry
    // was erased; the writer deletes a cancelled file once it lands. Both are
    // bounded by the number of outstanding writes.
    std::set<std::string> in_flight_;
    std::set<std::string> cancelled_;

    std::atomic<size_t> bytes_on_disk_{0};
    std::atomic<size_t> entries_on_disk_{0};
};

} // namespace server_prompt_disk
