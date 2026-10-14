#!/usr/bin/env python3
"""Guard the automatic disk swap tier wiring.

The tier moves serialized prompt-cache state out of RAM and must therefore fail
closed on any identity or integrity problem, and must never leave a file behind
when an entry is dropped. These are structural invariants that are easy to lose
in a refactor, so they are asserted against the sources.
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

    require('{"--cache-disk"}, "PATH"' in args, "--cache-disk option must take a PATH")
    require('{"--cache-disk-mib"}, "N"' in args, "--cache-disk-mib option must take a size")
    require('"LLAMA_ARG_CACHE_DISK"' in args, "--cache-disk must expose LLAMA_ARG_CACHE_DISK")
    require('"LLAMA_ARG_CACHE_DISK_MIB"' in args, "--cache-disk-mib must expose LLAMA_ARG_CACHE_DISK_MIB")
    require(
        "fs_is_directory(params.cache_disk_path)" in args,
        "--cache-disk must reject a path that is not a directory",
    )

    require("int32_t cache_disk_mib" in common_h, "common_params must carry cache_disk_mib")
    require("std::string cache_disk_path" in common_h, "common_params must carry cache_disk_path")

    server_cmake = read("tools/server/CMakeLists.txt")
    library_start = server_cmake.find("add_library(${TARGET} STATIC")
    library_end = server_cmake.find(")", library_start)
    require(library_start >= 0 and library_end > library_start, "server-context library block not found")
    library = server_cmake[library_start:library_end]
    require("server-prompt-disk.cpp" in library, "server-prompt-disk.cpp must build into server-context")
    require("server-prompt-disk.h" in library, "server-prompt-disk.h must be part of server-context")

    store = read("tools/server/server-prompt-disk.cpp")

    # atomic publish: a complete temporary file is renamed into place
    require('".tmp"' in store, "spill must stage through a temporary file")
    require("std::filesystem::rename(" in store, "spill must publish with an atomic rename")
    require('".swap"' in store, "published files must use the .swap suffix")

    # fail closed on every read
    require("h.identity != identity_" in store, "disk reads must validate the identity fingerprint")
    require("payload_hash != h.payload_hash" in store, "disk reads must validate the payload hash")
    require("decode_header(head, sizeof(head), h)" in store, "disk reads must validate the file header")
    require("non-contiguous sections" in store, "disk reads must validate the section layout")
    require("trailing bytes" in store, "disk reads must reject trailing data")

    # an invalidated entry keeps no payload, and the directory is process-owned
    require("e = entry {};" in store, "erase must leave the entry with no payload")
    require(store.count("wipe();") >= 2, "the store must wipe its directory at construction and destruction")

    # deferred writes: a published entry drains the writer instead of failing, and
    # an in-flight write whose entry is erased must not leave a file behind
    require("bool async_writes = true;" in read("tools/server/server-prompt-disk.h"),
            "the store must expose the deferred-write switch")
    require("void store::writer_loop()" in store, "deferred writes need a writer thread")
    require("bool store::submit(job && j)" in store, "spills must be submittable to the writer")
    require("cv_done_.wait(" in store, "flush must wait for the writer to go idle")
    require("cancelled_.erase(" in store, "the writer must delete a file whose entry was erased")
    require("flush();" in store, "open must drain a not-yet-written entry")

    task = read("tools/server/server-task.cpp")

    require("demote_oldest_ram" in task, "RAM pressure must demote through demote_oldest_ram")
    require(
        "states.pop_front" not in task,
        "evictions must go through erase_iterator so disk files are deleted",
    )
    require("disk.open(it_best->disk)" in task, "load must materialize a disk entry")
    require("restore_disk_successes" in task, "load must record disk-sourced restores")
    require("checkpoints_match_prompt" in task, "load must validate rehydrated checkpoint metadata")
    require("std::move(data.main)" in task, "demotion must hand the payload to the disk tier without copying")

    # metrics
    require("prompt_cache_disk_spill_total" in task, "Prometheus must export disk spill totals")
    require("prompt_cache_disk_restore_total" in task, "Prometheus must export disk restore totals")
    require("prompt_cache_disk_entries" in task, "Prometheus must export the disk entry gauge")
    require("prompt_cache_disk_bytes" in task, "Prometheus must export the disk byte gauge")

    context = read("tools/server/server-context.cpp")

    require("prompt_disk_identity" in context, "the disk tier must be keyed by a layout identity")
    require(
        "--cache-disk requires --cache-ram" in context,
        "the disk tier must refuse to run without the RAM prompt cache",
    )
    require('"disk_restore_committed"' in context, "a disk restore must be reason-coded")
    require('prompt_cache_source = "disk"' in context, "a disk restore must be labelled as such")

    common_cpp = read("tools/server/server-common.cpp")
    require("cache_disk_save_ms" in common_cpp, "timings must expose cache_disk_save_ms")
    require("cache_disk_load_ms" in common_cpp, "timings must expose cache_disk_load_ms")

    # user-facing documentation must cover the tier
    require("--cache-disk" in read("docs/beellama-args.md"), "beellama-args must document --cache-disk")
    require("--cache-disk-mib" in read("docs/beellama-args.md"), "beellama-args must document --cache-disk-mib")
    require("--cache-disk" in read("docs/beellama-features.md"), "beellama-features must document the disk tier")
    server_readme = read("tools/server/README.md")
    require("--cache-disk" in server_readme, "the server README must document --cache-disk")
    require("LLAMA_ARG_CACHE_DISK" in server_readme, "the server README must document the env var")

    print("cache disk static check passed")


if __name__ == "__main__":
    main()
