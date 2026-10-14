// See diskcache.hh (#437).
//
// The cache directory holds:
//   index.tsv    one line per entry, tab separated:
//                file, n, kv_bytes, cp_bytes, blob_bytes, used,
//                points (comma separated), tokens (comma separated),
//                image digests (comma separated, hex)
//   <file>       a 24-byte header (magic, version, n, blob bytes) then the blob
//   <file>.tmp   a write in flight (renamed on completion)
//
// A text index of a handful of conversations does not need a database, and the
// tokens are in it on purpose: the lookup then uses exactly the in-RAM store's
// prefix rule, so a request that matches an *inner* checkpoint of a saved
// conversation restores it (a hash of the full sequence could not).
#include "model/diskcache.hh"

#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace omph::model {

namespace {

constexpr size_t kIndexHeaderBytes = 4 + 4;  // magic, version, then one line per entry

std::string hex_of(const std::vector<int32_t> & tokens) {
    // a short, stable key for the file name (not a correctness check: the index
    // carries the tokens themselves)
    uint64_t h = 0xcbf29ce484222325ull;
    for (const int32_t t : tokens) {
        h = (h ^ (uint32_t) t) * 0x100000001b3ull;
        h ^= h >> 29;
    }
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return std::string(buf).substr(0, 16);
}

std::string join(const std::vector<int64_t> & v, const char comma) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i != 0) {
            out.push_back(comma);
        }
        out += std::to_string(v[i]);
    }
    return out;
}

std::vector<int64_t> split_i64(const std::string & s, const char comma) {
    std::vector<int64_t> out;
    std::string item;
    std::istringstream in(s);
    while (std::getline(in, item, comma)) {
        if (!item.empty()) {
            out.push_back(std::strtoll(item.c_str(), nullptr, 10));
        }
    }
    return out;
}

std::vector<int32_t> split_i32(const std::string & s) {
    std::vector<int32_t> out;
    std::string item;
    std::istringstream in(s);
    while (std::getline(in, item, ',')) {
        if (!item.empty()) {
            out.push_back((int32_t) std::strtol(item.c_str(), nullptr, 10));
        }
    }
    return out;
}

std::vector<std::string> split_str(const std::string & s) {
    std::vector<std::string> out;
    std::string item;
    std::istringstream in(s);
    while (std::getline(in, item, ',')) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

// FNV-1a over a byte range: 8 bytes per step at ~1 byte per cycle, so a 3 GiB
// blob costs about a second - the price of never restoring a corrupted one.
uint64_t fnv1a(const void * data, const size_t bytes) {
    const auto * p = static_cast<const uint8_t *>(data);
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < bytes; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

bool read_all(const int fd, void * dst, const size_t bytes) {
    auto * p = static_cast<uint8_t *>(dst);
    size_t off = 0;
    while (off < bytes) {
        const ssize_t n = ::read(fd, p + off, bytes - off);
        if (n <= 0) {
            return false;
        }
        off += (size_t) n;
    }
    return true;
}

bool write_all(const int fd, const void * src, const size_t bytes) {
    const auto * p = static_cast<const uint8_t *>(src);
    size_t off = 0;
    while (off < bytes) {
        const ssize_t n = ::write(fd, p + off, bytes - off);
        if (n <= 0) {
            return false;
        }
        off += (size_t) n;
    }
    return true;
}

void put_u32(uint8_t * p, const uint32_t x) {
    for (int i = 0; i < 4; ++i) {
        p[i] = (uint8_t) (x >> (8 * i));
    }
}
void put_u64(uint8_t * p, const uint64_t x) {
    for (int i = 0; i < 8; ++i) {
        p[i] = (uint8_t) (x >> (8 * i));
    }
}
uint32_t get_u32(const uint8_t * p) {
    return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24;
}
uint64_t get_u64(const uint8_t * p) {
    uint64_t x = 0;
    for (int i = 0; i < 8; ++i) {
        x |= (uint64_t) p[i] << (8 * i);
    }
    return x;
}

} // namespace

std::string disk_file_name(const std::string & model_key, const int64_t n, const std::vector<int32_t> & tokens) {
    return "omph-" + model_key.substr(0, 8) + "-" + std::to_string(n) + "-" + hex_of(tokens) + ".kvc";
}

DiskCache::~DiskCache() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
    if (fd_ >= 0) {
        (void) ::close(fd_);
    }
}

bool DiskCache::open(const std::string & dir, const std::string & model_key, const int64_t budget_mib) {
    if (dir.empty() || budget_mib <= 0) {
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    fd_ = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd_ < 0) {
        std::fprintf(stderr, "cache dir %s: %s\n", dir.c_str(), std::strerror(errno));
        return false;
    }
    dir_ = dir;
    model_key_ = model_key;
    budget_ = (size_t) budget_mib << 20;
    (void) read_index();
    const size_t before = index_.size();
    const std::string prefix = "omph-" + model_key.substr(0, 8) + "-";
    index_.erase(std::remove_if(index_.begin(), index_.end(),
                                [&](const Entry & e) { return e.file.rfind(prefix, 0) != 0; }),
                 index_.end());
    if (index_.size() != before) {
        total_ = 0;
        for (const Entry & e : index_) {
            total_ += e.blob_bytes;
        }
        (void) write_index();
    }
    if (std::getenv("OMPH_DISK_DEBUG") != nullptr) {
        std::fprintf(stderr, "diskcache: %s, %zu entries, %.1f MiB\n", dir_.c_str(), index_.size(),
                     (double) total_ / 1048576.0);
        for (const Entry & e : index_) {
            std::fprintf(stderr, "  %s n=%lld points=%zu tokens=%zu\n", e.file.c_str(), (long long) e.n,
                         e.points.size(), e.tokens.size());
        }
    }
    thread_ = std::thread([this] { worker(); });
    return true;
}

size_t DiskCache::entries() const {
    std::lock_guard<std::mutex> lock(mu_);
    return index_.size();
}

int64_t DiskCache::total_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_;
}

const DiskCache::Entry * DiskCache::match(const std::function<bool(const Entry &, int64_t)> & fits) {
    if (!enabled()) {
        return nullptr;
    }
    const bool dbg = std::getenv("OMPH_DISK_DEBUG") != nullptr;
    std::lock_guard<std::mutex> lock(mu_);
    if (dbg) {
        std::fprintf(stderr, "diskcache: match over %zu entries\n", index_.size());
    }
    Entry * best = nullptr;
    int64_t best_p = -1;
    for (Entry & e : index_) {
        for (const int64_t p : e.points) {
            if (p <= best_p) {
                continue;
            }
            if (fits(e, p)) {
                best = &e;
                best_p = p;
                if (dbg) {
                    std::fprintf(stderr, "diskcache: %s matches at point %lld\n", e.file.c_str(), (long long) p);
                }
            } else if (dbg) {
                std::fprintf(stderr, "diskcache: %s point %lld no\n", e.file.c_str(), (long long) p);
            }
        }
    }
    if (best != nullptr) {
        best->used = ++clock_;
    }
    return best;
}

bool DiskCache::load(const Entry & e, std::vector<uint8_t> & blob) {
    const int fd = ::openat(fd_, e.file.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    uint8_t header[kHeaderBytes];
    bool ok = read_all(fd, header, sizeof(header)) && get_u32(header) == kMagic &&
              get_u32(header + 4) == kVersion && (int64_t) get_u64(header + 8) == e.n &&
              (int64_t) get_u64(header + 16) == e.blob_bytes && get_u64(header + 24) == e.blob_hash;
    if (ok) {
        blob.resize((size_t) e.blob_bytes);
        ok = read_all(fd, blob.data(), blob.size()) && fnv1a(blob.data(), blob.size()) == e.blob_hash;
    }
    (void) ::close(fd);
    return ok;
}

void DiskCache::save(const Entry & e, const void * src, const int64_t bytes, std::atomic<bool> * done) {
    if (!enabled() || src == nullptr || bytes <= 0) {
        if (done != nullptr) {
            done->store(true);
        }
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.size() >= 2) {
        if (done != nullptr) {
            done->store(true);  // no room: the file simply stays older
        }
        return;
    }
    Entry copy = e;
    copy.blob_bytes = bytes;
    copy.blob_hash = fnv1a(src, (size_t) bytes);  // hashed here, verified on load
    queue_.push_back(Job{copy, src, bytes, done});
    cv_.notify_one();
}

bool DiskCache::write_job(const Job & j) {
    const std::string path = dir_ + "/" + j.e.file;
    const std::string tmp = path + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return false;
    }
    uint8_t header[kHeaderBytes];
    put_u32(header, kMagic);
    put_u32(header + 4, kVersion);
    put_u64(header + 8, (uint64_t) j.e.n);
    put_u64(header + 16, (uint64_t) j.bytes);
    put_u64(header + 24, j.e.blob_hash);
    const bool ok = write_all(fd, header, sizeof(header)) && write_all(fd, j.src, (size_t) j.bytes);
    (void) ::close(fd);
    if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) {
        (void) ::unlink(tmp.c_str());
        return false;
    }
    return true;
}

void DiskCache::worker() {
    for (;;) {
        Job j;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
                if (stop_) {
                    return;
                }
                continue;
            }
            j = queue_.front();
            queue_.pop_front();
        }
        const bool ok = write_job(j);
        if (ok) {
            std::lock_guard<std::mutex> lock(mu_);
            bool found = false;
            for (Entry & e : index_) {
                if (e.file == j.e.file) {
                    total_ += j.e.blob_bytes - e.blob_bytes;
                    e = j.e;
                    e.used = ++clock_;
                    found = true;
                    break;
                }
            }
            if (!found) {
                Entry e = j.e;
                e.used = ++clock_;
                index_.push_back(e);
                total_ += e.blob_bytes;
            }
            while (total_ > (int64_t) budget_ && !index_.empty()) {
                auto lru = std::min_element(index_.begin(), index_.end(),
                                            [](const Entry & a, const Entry & b) { return a.used < b.used; });
                const std::string victim = lru->file;
                total_ -= lru->blob_bytes;
                index_.erase(lru);
                (void) ::unlink((dir_ + "/" + victim).c_str());
            }
            (void) write_index();
        }
        if (j.done != nullptr) {
            j.done->store(true);
        }
    }
}

std::string DiskCache::format_line(const Entry & e) {
    std::string line;
    line += e.file;
    line += '\t' + std::to_string(e.n);
    line += '\t' + std::to_string(e.kv_bytes);
    line += '\t' + std::to_string(e.cp_bytes);
    line += '\t' + std::to_string(e.blob_bytes);
    line += '\t' + std::to_string(e.blob_hash);
    line += '\t' + std::to_string(e.used);
    line += '\t' + join(e.points, ',');
    line += '\t';
    for (size_t i = 0; i < e.tokens.size(); ++i) {
        if (i != 0) {
            line.push_back(',');
        }
        line += std::to_string(e.tokens[i]);
    }
    line += '\t';
    for (size_t i = 0; i < e.images.size(); ++i) {
        if (i != 0) {
            line.push_back(',');
        }
        line += e.images[i];
    }
    return line;
}

bool DiskCache::read_index() {
    const int fd = ::openat(fd_, "index.tsv", O_RDONLY);
    if (fd < 0) {
        return false;
    }
    std::string text;
    char buf[65536];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n < 0) {
            (void) ::close(fd);
            return false;
        }
        if (n == 0) {
            break;
        }
        text.append(buf, (size_t) n);
    }
    (void) ::close(fd);
    if (std::getenv("OMPH_DISK_DEBUG") != nullptr) {
        std::fprintf(stderr, "diskcache: index %zu bytes\n", text.size());
    }
    std::istringstream in(text);
    std::string line;
    (void) std::getline(in, line);  // the header line
    if (std::getenv("OMPH_DISK_DEBUG") != nullptr) {
        std::fprintf(stderr, "diskcache: header '%s'\n", line.c_str());
    }
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> f;
        std::string item;
        std::istringstream ls(line);
        while (std::getline(ls, item, '\t')) {
            f.push_back(item);
        }
        // the last field (the image digests) is empty for a text-only
        // conversation and getline does not return a trailing empty field
        if (f.size() < 9) {
            continue;  // a truncated tail: skip the line
        }
        Entry e;
        e.file = f[0];
        e.n = std::strtoll(f[1].c_str(), nullptr, 10);
        e.kv_bytes = std::strtoll(f[2].c_str(), nullptr, 10);
        e.cp_bytes = std::strtoll(f[3].c_str(), nullptr, 10);
        e.blob_bytes = std::strtoll(f[4].c_str(), nullptr, 10);
        e.blob_hash = std::strtoull(f[5].c_str(), nullptr, 10);
        e.used = (uint64_t) std::strtoll(f[6].c_str(), nullptr, 10);
        e.points = split_i64(f[7], ',');
        e.tokens = split_i32(f[8]);
        if (f.size() > 9) {
            e.images = split_str(f[9]);
        }
        if (e.n <= 0 || e.file.empty() || (int64_t) e.tokens.size() < e.n) {
            if (std::getenv("OMPH_DISK_DEBUG") != nullptr) {
                std::fprintf(stderr, "diskcache: skipping %s n=%lld tokens=%zu\n", e.file.c_str(),
                             (long long) e.n, e.tokens.size());
            }
            continue;  // not one of ours
        }
        total_ += e.blob_bytes;
        clock_ = std::max(clock_, e.used);
        index_.push_back(std::move(e));
    }
    return true;
}

bool DiskCache::write_index() {
    const std::string path = dir_ + "/index.tsv";
    const std::string tmp = path + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return false;
    }
    std::string out = "omph-cache\t" + std::to_string(kVersion) + "\n";
    for (const Entry & e : index_) {
        out += format_line(e);
        out.push_back('\n');
    }
    const bool ok = write_all(fd, out.data(), out.size());
    (void) ::close(fd);
    if (!ok) {
        (void) ::unlink(tmp.c_str());
        return false;
    }
    return ::rename(tmp.c_str(), path.c_str()) == 0;
}

} // namespace omph::model
