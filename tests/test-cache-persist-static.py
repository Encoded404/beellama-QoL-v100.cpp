#!/usr/bin/env python3
"""Guard the persistent prompt-cache tier wiring.

The persistent tier keeps serialized prompt-derived KV state across process
restarts, so every guard the session swap tier has must hold here too, plus the
restart-specific ones: no payload rewrite on promotion, an atomically published
sidecar index, and a startup pass that deletes anything it cannot fully
validate. These are structural invariants that are easy to lose in a refactor,
so they are asserted against the sources.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def main() -> None:
    args = read("common/arg.cpp")
    common_h = read("common/common.h")

    require('{"--cache-persist"}, "PATH"' in args, "--cache-persist option must take a PATH")
    require('"LLAMA_ARG_CACHE_PERSIST"' in args, "--cache-persist must expose LLAMA_ARG_CACHE_PERSIST")
    require(
        "fs_is_directory(params.cache_persist_path)" in args,
        "--cache-persist must reject a path that is not a directory",
    )
    require(
        "std::string cache_persist_path" in common_h,
        "common_params must carry cache_persist_path",
    )

    server_cmake = read("tools/server/CMakeLists.txt")
    library_start = server_cmake.find("add_library(${TARGET} STATIC")
    library_end = server_cmake.find(")", library_start)
    require(library_start >= 0 and library_end > library_start, "server-context library block not found")
    library = server_cmake[library_start:library_end]
    require("server-prompt-persist.cpp" in library, "server-prompt-persist.cpp must build into server-context")
    require("server-prompt-persist.h" in library, "server-prompt-persist.h must be part of server-context")

    persist = read("tools/server/server-prompt-persist.cpp")
    persist_h = read("tools/server/server-prompt-persist.h")

    # the sidecar index is published atomically: staged temporary file + rename
    require('"index.bin.tmp"' in persist, "the index must stage through a temporary file")
    require(
        "std::filesystem::rename(" in persist,
        "the index must publish with an atomic rename",
    )
    require("INDEX_MAGIC" in persist_h and "INDEX_VERSION" in persist_h, "the index format must carry magic and version")

    # a stale scan clears the tier instead of trusting unparsable state
    require(
        "remove_stale_files" in persist and "remove_stale_files(dir, {})" in read("tools/server/server-task.cpp"),
        "a missing or corrupt index must clear the directory, not adopt files blindly",
    )

    task = read("tools/server/server-task.cpp")

    # promotion moves files; it never rewrites a payload
    require(
        "move_file(src, dst)" in task and "patch_header_identity(dst" in task,
        "promotion must move the swap file and re-stamp only its header",
    )
    require(
        task.find("patch_header_identity") < task.find("persist.adopt(e)", task.find("patch_header_identity")),
        "the header must be re-stamped before adoption validates the identity",
    )
    require(
        "bool move_file(" in persist,
        "cross-filesystem moves must fall back to a full copy",
    )

    # the persistent store must not wipe its directory at either end of a run,
    # and the cache must not erase persist-entry files at teardown
    store = read("tools/server/server-prompt-disk.cpp")
    require("cfg_.persist" in store, "the store must switch wipe behaviour by config")
    require(
        store.count("cfg_.persist") >= 3,
        "construction, wipe and destruction must all key off the persist flag",
    )
    disk_h = read("tools/server/server-prompt-disk.h")
    require("bool persist_tier" in disk_h, "cache entries must record which tier owns their file")
    require(
        "!state.disk.persist_tier" in task,
        "the cache destructor must not delete files owned by the persistent tier",
    )
    require(
        "it_best->disk.persist_tier" in task,
        "entry opens must dispatch by owning tier",
    )
    require(
        "persist_tier ? persist : disk" in task,
        "entry erases must dispatch by owning tier",
    )

    # the identity must distinguish same-name/different-content models
    context = read("tools/server/server-context.cpp")
    require("prompt_persist_identity" in context, "the persist tier must compute its own identity")
    require(
        "file_fingerprint(params.model.path)" in context,
        "the persist identity must fingerprint the model file content",
    )
    require(
        "params_base.cache_disk_path == params_base.cache_persist_path" in context,
        "--cache-persist must be refused when it equals --cache-disk",
    )
    require(
        "persist_load(!params_base.mmproj.path.empty())" in context,
        "media-token entries must require the mmproj of the next run",
    )
    require(
        "prompt_cache->persist_save();" in context,
        "the shutdown path must call persist_save before releasing the cache",
    )

    # the destructors: the persistent store keeps its directory, the cache
    # keeps persist-owned files
    require("wipe();" in read("tools/server/server-prompt-disk.cpp"), "the session store must still wipe itself")


if __name__ == "__main__":
    main()