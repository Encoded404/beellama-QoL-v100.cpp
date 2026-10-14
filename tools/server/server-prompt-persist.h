#pragma once

// Persistent prompt-cache tier helpers, used by server_prompt_cache.
//
// The tier stores exactly the self-contained serialized sequence state the
// automatic disk swap tier produces (same file format, same validation), plus
// one small sidecar index that carries everything that only lives in RAM:
//
//   - the entry order (the cache's list order, oldest first),
//   - the serialized token sequence of each entry (server_tokens::serialize),
//     which is what prompt-prefix matching needs and what survives nothing on
//     a restart today,
//   - the checkpoint appendix metadata, so checkpoint reuse still works after
//     a restart without loading any payload.
//
// Promotion of an entry that was already spilled to the session swap tier is
// intentionally a filesystem rename plus an in-place 128-byte header identity
// patch. The payload sections and the payload hash are never read or rewritten.

#include "server-prompt-disk.h"

#include <cstdint>
#include <string>
#include <vector>

namespace server_prompt_persist {

constexpr uint32_t INDEX_MAGIC   = 0x50494350; // 'PCIP'
constexpr uint32_t INDEX_VERSION = 1;

// minimum bytes of one index record: name (2) + flags (1) + tokens length (4)
// + one token + checkpoint count (4)
constexpr size_t INDEX_RECORD_MIN = 2 + 1 + 4 + 4 + 4;

struct ckpt_meta {
    int64_t   n_tokens = 0;
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;
};

struct record {
    // file name inside the persist directory, e.g. "pcp-3.swap"
    std::string name;

    // server_tokens::serialize() payload of the owning cache entry
    std::vector<char> tokens;

    // the serialized tokens contain media chunks (mmproj required to restore)
    bool has_media = false;

    // checkpoint appendix metadata, buffers stay in the entry file
    std::vector<ckpt_meta> checkpoints;
};

// Read `index.bin` from dir. A missing, corrupt or foreign index is a clean
// miss: returns false and leaves records empty, so the caller clears the
// directory instead of trusting anything in it.
bool index_read(const std::string & dir, std::vector<record> & records);

// Atomically publish `index.bin` (temporary file + rename). The tier identity
// is stored for diagnostics. Returns false on I/O failure; the caller keeps
// the previous index in that case.
bool index_write(const std::string & dir, uint64_t identity, const std::vector<record> & records);

// Remove every `pcp-*.swap` / `pcp-*.tmp` / `index.bin.tmp` in dir whose name
// is not referenced by `keep`. Returns the number of removed files.
size_t remove_stale_files(const std::string & dir, const std::vector<record> & keep);

// Delete a single file, ignoring absence.
void remove_file(const std::string & path);

// Content fingerprint of a model file: the file size, the last-write time and
// FNV-1a over fixed slices spread across the file. Stable across restarts with
// the same file. The mtime separates a re-finetune that keeps the same name
// but holds different content; the slices keep copies of identical content
// sharing a fingerprint, and catch in-place content mutations that preserve
// the mtime when they touch a sampled region. Returns 0 when the path is
// empty or not a readable regular file, so the identity then relies on model
// metadata and tensor sizes only.
uint64_t file_fingerprint(const std::string & path);

// Move a file to another directory: rename when it lands, otherwise a full
// copy followed by removing the source (only used when the persist directory
// is on a different filesystem than the swap tier). Returns false when the
// file could not be moved; the source is left intact then.
bool move_file(const std::string & src, const std::string & dst);

// Patch the identity stored in a published entry header (the first 128 bytes,
// in place). The payload sections and the payload hash stay untouched. Used
// after promoting a swap file into the persist directory, so the persistent
// tier's read path accepts a file the swap tier wrote.
bool patch_header_identity(const std::string & path, uint64_t identity);

} // namespace server_prompt_persist