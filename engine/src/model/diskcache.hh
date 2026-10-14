// The saved conversations mirrored to disk (#437): a restart resumes a long
// context with a read instead of a prefill.
//
// One file per conversation plus a text index (`index.tsv`). An entry carries
// the tokens and the image digests it was saved for, so a lookup uses the same
// prefix rule as the in-RAM store (`prefix_equal`, supplied by the Generator
// through `match`): a foreign or stale file can only be discarded.
//
// The blob itself is opaque here (the Generator's snapshot layout: the KV
// prefix, then one checkpoint per point); a small header records n and the
// length so garbage is rejected.
//
// `save` hands the bytes to a writer thread and never copies them: the caller
// keeps the buffer alive until the job's `done` flag is set (the Generator's
// eviction skips snapshots whose flag is set). A failed or dropped write only
// costs a future prefill.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace omph::model {

class DiskCache {
public:
    static constexpr uint32_t kMagic = 0x31434d4fu;  // "OMC1"
    static constexpr uint32_t kVersion = 3;
    static constexpr size_t kHeaderBytes = 4 + 4 + 8 + 8 + 8;  // magic, version, n, bytes, FNV-1a 64

    struct Entry {
        int64_t n = 0;                     // positions the entry covers
        std::string file;                  // its name inside the cache directory
        std::vector<int32_t> tokens;       // the saved sequence's ids (n of them)
        std::vector<std::string> images;   // the image digests, in order (#363)
        int64_t kv_bytes = 0;              // the KV prefix's bytes in the blob
        int64_t cp_bytes = 0;              // one checkpoint's bytes
        int64_t blob_bytes = 0;
        uint64_t blob_hash = 0;            // FNV-1a of the blob: a corrupted file is discarded, not used
        std::vector<int64_t> points;       // ascending
        uint64_t used = 0;                 // LRU clock, persisted
    };

    DiskCache() = default;
    DiskCache(const DiskCache &) = delete;
    DiskCache & operator=(const DiskCache &) = delete;
    ~DiskCache();

    // Opens (creating it) `dir`, reads the index, drops the entries of another
    // model and evicts to fit `budget_mib`. An empty `dir` disables it.
    bool open(const std::string & dir, const std::string & model_key, int64_t budget_mib);
    bool enabled() const { return fd_ >= 0; }
    // The best entry (its longest point) for which `fits(entry, position)` is
    // true. The callback is the Generator's prefix rule.
    const Entry * match(const std::function<bool(const Entry &, int64_t)> & fits);
    // The blob of an entry; the header is checked against the entry again.
    bool load(const Entry & e, std::vector<uint8_t> & blob);
    // Queues a write of `bytes` at `src` (the caller keeps it alive until `done`
    // is set). A full queue drops the job: the file simply stays older.
    void save(const Entry & e, const void * src, int64_t bytes, std::atomic<bool> * done);
    size_t entries() const;
    int64_t total_bytes() const;

private:
    struct Job {
        Entry e;
        const void * src = nullptr;
        int64_t bytes = 0;
        std::atomic<bool> * done = nullptr;
    };
    void worker();
    bool write_job(const Job & j);
    static std::string format_line(const Entry & e);
    bool read_index();
    bool write_index();
    int fd_ = -1;  // the directory, for openat
    std::string dir_;
    std::string model_key_;
    size_t budget_ = 0;
    std::vector<Entry> index_;
    int64_t total_ = 0;
    uint64_t clock_ = 0;
    mutable std::mutex mu_;
    std::deque<Job> queue_;
    bool stop_ = false;
    std::thread thread_;
    std::condition_variable cv_;
};

// "omph-<model8>-<n>-<hash16>.kvc": a name unique per (model, positions, tokens).
std::string disk_file_name(const std::string & model_key, int64_t n, const std::vector<int32_t> & tokens);

} // namespace omph::model
