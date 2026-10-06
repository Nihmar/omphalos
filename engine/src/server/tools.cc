#include "server/tools.hh"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace omph::server::tools {
namespace {

namespace fs = std::filesystem;
using Json = omph::text::Json;

// llama.cpp's caps, so its UI and prompts behave the same (server-tools.cpp).
constexpr size_t kReadMax = 16 * 1024;          // read_file's text budget
constexpr size_t kReadMaxBase64 = 32 * 1024 * 1024;
constexpr size_t kExecMaxOutput = 16 * 1024;
constexpr int kExecDefaultTimeout = 10;
constexpr int kExecMaxTimeout = 60;
constexpr size_t kSearchMaxResults = 100;
constexpr size_t kListDefaultLimit = 100;
constexpr size_t kListMaxLimit = 100;
constexpr size_t kWalkCap = 200000;  // a walk that finds nothing must end
constexpr int kListTimeout = 15;

Json str(const std::string & s) { return Json::string(s); }
Json boolean(const bool b) { return Json::boolean(b); }

Json error(const std::string & message) {
    Json j = Json::object();
    j.set("error", str(message));
    return j;
}

Json plain(const std::string & text) {
    Json j = Json::object();
    j.set("plain_text_response", str(text));
    return j;
}

//
// parameters
//

const Json * param(const Json & params, const char * key) {
    return params.is_object() ? params.find(key) : nullptr;
}

std::string string_param(const Json & params, const char * key, const std::string & fallback = {}) {
    const Json * v = param(params, key);
    return v != nullptr && v->is_string() ? v->as_string() : fallback;
}

int64_t int_param(const Json & params, const char * key, const int64_t fallback) {
    const Json * v = param(params, key);
    return v != nullptr && v->is_number() ? (int64_t) v->as_number() : fallback;
}

bool bool_param(const Json & params, const char * key, const bool fallback = false) {
    const Json * v = param(params, key);
    return v != nullptr && v->is_bool() ? v->as_bool() : fallback;
}

// A required string: the callers report the missing name themselves, so the
// messages stay in the tool's vocabulary ("path is required").
bool required_string(const Json & params, const char * key, std::string & out, std::string & error_out) {
    const Json * v = param(params, key);
    if (v == nullptr || !v->is_string()) {
        error_out = std::string(key) + " is required and must be a string";
        return false;
    }
    out = v->as_string();
    return true;
}

//
// paths
//

fs::path base_dir(const std::string & cwd) {
    std::error_code ec;
    const fs::path p = cwd.empty() ? fs::current_path(ec) : fs::path(cwd);
    const fs::path abs = fs::absolute(p, ec);
    return (ec ? p : abs).lexically_normal();
}

fs::path resolve(const std::string & cwd, const std::string & path) {
    fs::path p(path);
    if (p.is_relative()) {
        p = base_dir(cwd) / p;
    }
    std::error_code ec;
    const fs::path abs = fs::absolute(p, ec);
    return (ec ? p : abs).lexically_normal();
}

std::string path_str(const fs::path & p) { return p.generic_string(); }

std::string rel_to(const fs::path & base, const fs::path & p) {
    std::error_code ec;
    const fs::path rel = fs::relative(p, base, ec);
    return ec ? path_str(p) : rel.generic_string();
}

bool read_file_bytes(const fs::path & p, std::string & out, const size_t max = (size_t) -1) {
    std::error_code ec;
    if (!fs::exists(p, ec) || !fs::is_regular_file(p, ec)) {
        return false;
    }
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    if (out.size() > max) {
        out.resize(max);
    }
    return true;
}

//
// subprocesses (shell commands and git listings)
//

struct ProcResult {
    std::string output;
    int exit_code = -1;
    bool timed_out = false;
    bool spawn_error = false;
};

ProcResult run_process(const std::vector<std::string> & argv, const std::string & cwd, const int timeout_s,
                       const size_t max_output, const bool append_all = true) {
    ProcResult res;
    int fds[2];
    if (pipe(fds) != 0) {
        res.spawn_error = true;
        return res;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        res.spawn_error = true;
        return res;
    }
    if (pid == 0) {  // the child: its own process group, so a timeout kills the tree
        setpgid(0, 0);
        const int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) {
            (void) dup2(devnull, STDIN_FILENO);
            close(devnull);
        }
        (void) dup2(fds[1], STDOUT_FILENO);
        (void) dup2(fds[1], STDERR_FILENO);
        close(fds[0]);
        close(fds[1]);
        if (!cwd.empty() && chdir(cwd.c_str()) != 0) {
            _exit(127);
        }
        std::vector<char *> args;
        args.reserve(argv.size() + 1);
        for (const std::string & a : argv) {
            args.push_back(const_cast<char *>(a.c_str()));
        }
        args.push_back(nullptr);
        execvp(args[0], args.data());
        _exit(127);
    }
    close(fds[1]);
    (void) fcntl(fds[0], F_SETFL, O_NONBLOCK);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    bool eof = false;
    bool truncated = false;
    while (!eof) {
        char buf[8192];
        const ssize_t n = read(fds[0], buf, sizeof buf);
        if (n > 0) {
            if (res.output.size() < max_output) {
                const size_t room = max_output - res.output.size();
                res.output.append(buf, std::min((size_t) n, room));
                truncated = truncated || (size_t) n > room;
            } else {
                truncated = true;
            }
        } else if (n == 0) {
            eof = true;
        } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            eof = true;
        }
        if (eof) {
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            res.timed_out = true;
            break;
        }
        if (n <= 0) {
            pollfd pfd{fds[0], POLLIN, 0};
            (void) poll(&pfd, 1, 20);
        }
    }
    if (res.timed_out) {
        (void) kill(-pid, SIGKILL);
        (void) kill(pid, SIGKILL);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    // whatever is still buffered in the closed pipe
    while (true) {
        char buf[8192];
        const ssize_t n = read(fds[0], buf, sizeof buf);
        if (n <= 0) {
            break;
        }
        if (res.output.size() < max_output) {
            const size_t room = max_output - res.output.size();
            res.output.append(buf, std::min((size_t) n, room));
            truncated = truncated || (size_t) n > room;
        } else {
            truncated = true;
        }
    }
    close(fds[0]);
    if (res.timed_out) {
        res.exit_code = -1;
    } else if (WIFEXITED(status)) {
        res.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        res.exit_code = 128 + WTERMSIG(status);
    }
    if (truncated && append_all) {
        res.output += "\n[output truncated]";
    }
    return res;
}

//
// listings (git ls-files with a walker fallback), globs
//

struct LsEntry {
    std::string rel;
    bool is_dir = false;
};

bool is_junk_dir(const std::string & name) {
    static const char * kJunk[] = {".git", "node_modules", "build", "dist", "target", "__pycache__",
                                   ".venv", "venv", ".cache", ".idea", ".vscode", ".next", "out", "bin", "obj"};
    for (const char * j : kJunk) {
        if (name == j) {
            return true;
        }
    }
    return false;
}

// The files git would track (respecting .gitignore); false when the directory
// is not inside a repository or git is not there.
bool git_files(const fs::path & base, std::vector<std::string> & out) {
    const ProcResult r = run_process({"git", "-C", path_str(base), "ls-files", "-z", "--cached", "--others",
                                      "--exclude-standard"},
                                     "", kListTimeout, 8 * 1024 * 1024, /*append_all=*/false);
    if (r.spawn_error || r.timed_out || r.exit_code != 0 || r.output.empty()) {
        return false;
    }
    size_t pos = 0;
    while (pos < r.output.size()) {
        const size_t nul = r.output.find('\0', pos);
        const std::string name = r.output.substr(pos, nul == std::string::npos ? std::string::npos : nul - pos);
        if (!name.empty()) {
            out.push_back(name);
        }
        if (nul == std::string::npos) {
            break;
        }
        pos = nul + 1;
    }
    return !out.empty();
}

void walk(const fs::path & base, std::vector<LsEntry> & out, bool & truncated) {
    std::error_code ec;
    fs::recursive_directory_iterator it(base, fs::directory_options::skip_permission_denied, ec);
    const fs::recursive_directory_iterator end;
    while (!ec && it != end) {
        if (out.size() >= kWalkCap) {
            truncated = true;
            break;
        }
        const fs::path p = it->path();
        const bool is_dir = it->is_directory(ec);
        if (is_dir && is_junk_dir(p.filename().string())) {
            it.disable_recursion_pending();
        } else {
            out.push_back({rel_to(base, p), is_dir});
        }
        it.increment(ec);
        if (ec) {
            ec.clear();
            break;
        }
    }
}

// The files under `base` plus their directory prefixes (git gives files only),
// or a walk when it is not a repository.
void list_entries(const fs::path & base, std::vector<LsEntry> & out, bool & truncated) {
    std::vector<std::string> files;
    if (git_files(base, files)) {
        std::set<std::string> dirs;
        for (const std::string & f : files) {
            out.push_back({f, false});
            fs::path parent = fs::path(f).parent_path();
            while (!parent.empty() && parent != ".") {
                dirs.insert(parent.generic_string());
                parent = parent.parent_path();
            }
        }
        for (const std::string & d : dirs) {
            out.push_back({d, true});
        }
        std::sort(out.begin(), out.end(), [](const LsEntry & a, const LsEntry & b) { return a.rel < b.rel; });
        return;
    }
    walk(base, out, truncated);
}

std::string glob_to_regex(const std::string & pattern) {
    std::string re;
    for (size_t i = 0; i < pattern.size(); ++i) {
        const char c = pattern[i];
        if (c == '*') {
            if (i + 1 < pattern.size() && pattern[i + 1] == '*') {
                ++i;
                if (i + 1 < pattern.size() && pattern[i + 1] == '/') {
                    re += "(?:.*/)?";  // "**/" also matches no directory at all
                    ++i;
                } else {
                    re += ".*";
                }
            } else {
                re += "[^/]*";
            }
        } else if (c == '?') {
            re += "[^/]";
        } else if (std::strchr(".^$|()[]{}+\\", c) != nullptr) {
            re += '\\';
            re += c;
        } else {
            re += c;
        }
    }
    return re;
}

bool glob_match(const std::string & rel, const std::string & pattern) {
    if (pattern.empty() || pattern == "**") {
        return true;
    }
    std::string pat = pattern;
    if (pat.find('/') == std::string::npos) {  // the basename at any depth
        return std::regex_match(fs::path(rel).filename().string(), std::regex(glob_to_regex(pat)));
    }
    if (pat.rfind("**/", 0) != 0 && pat.front() != '/') {
        pat = "**/" + pat;
    }
    if (pat.front() == '/') {
        pat.erase(0, 1);
    }
    return std::regex_match(rel, std::regex(glob_to_regex(pat)));
}

std::string base64_encode(const std::string & in) {
    static const char kTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    for (size_t i = 0; i < in.size(); i += 3) {
        const uint32_t v = ((uint32_t) (unsigned char) in[i] << 16) |
                           (i + 1 < in.size() ? (uint32_t) (unsigned char) in[i + 1] << 8 : 0) |
                           (i + 2 < in.size() ? (uint32_t) (unsigned char) in[i + 2] : 0);
        out += kTable[(v >> 18) & 63];
        out += kTable[(v >> 12) & 63];
        out += i + 1 < in.size() ? kTable[(v >> 6) & 63] : '=';
        out += i + 2 < in.size() ? kTable[v & 63] : '=';
    }
    return out;
}

std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> lines;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

//
// the tools
//

Json tool_read_file(const Json & params, const std::string & cwd) {
    std::string path;
    std::string missing;
    if (!required_string(params, "path", path, missing)) {
        return error(missing);
    }
    const int64_t start_line = std::max<int64_t>(1, int_param(params, "start_line", 1));
    const int64_t end_line = int_param(params, "end_line", -1);
    const bool append_loc = bool_param(params, "append_loc", false);
    const bool as_base64 = string_param(params, "resp_type") == "base64";  // from x-resp-type
    const fs::path p = resolve(cwd, path);
    std::error_code ec;
    if (!fs::exists(p, ec) || !fs::is_regular_file(p, ec)) {
        return error("cannot stat file: " + path);
    }
    const uintmax_t size = fs::file_size(p, ec);

    if (as_base64) {
        if (size > kReadMaxBase64) {
            return error("file too large (" + std::to_string((size_t) size) + " bytes, max " +
                         std::to_string(kReadMaxBase64) + ")");
        }
        std::string bytes;
        if (!read_file_bytes(p, bytes)) {
            return error("failed to open file: " + path);
        }
        Json j = Json::object();
        j.set("base64", str(base64_encode(bytes)));
        j.set("size_bytes", Json::integer((int64_t) bytes.size()));
        return j;
    }
    if (size > kReadMax && end_line == -1) {
        return error("file too large (" + std::to_string((size_t) size) + " bytes, max " +
                     std::to_string(kReadMax) + "). Use start_line/end_line to read a portion.");
    }
    std::string content;
    if (!read_file_bytes(p, content)) {
        return error("failed to open file: " + path);
    }
    const std::vector<std::string> lines = split_lines(content);
    std::string out;
    for (int64_t i = start_line - 1; i < (int64_t) lines.size(); i++) {
        if (end_line != -1 && i >= end_line) {
            break;
        }
        const std::string & line = lines[i];
        const std::string out_line = append_loc ? std::to_string(i + 1) + "\u2192" + line + "\n" : line + "\n";
        if (out.size() + out_line.size() > kReadMax) {
            out += "[output truncated]";
            break;
        }
        out += out_line;
    }
    return plain(out);
}

Json tool_write_file(const Json & params, const std::string & cwd) {
    std::string path;
    std::string missing;
    if (!required_string(params, "path", path, missing)) {
        return error(missing);
    }
    std::string content;
    if (!required_string(params, "content", content, missing)) {
        return error(missing);
    }
    const fs::path p = resolve(cwd, path);
    std::error_code ec;
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path(), ec);
    }
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) {
        return error("failed to write file: " + path);
    }
    out.write(content.data(), (std::streamsize) content.size());
    out.close();
    if (!out) {
        return error("failed to write file: " + path);
    }
    Json j = Json::object();
    j.set("result", str("file written successfully"));
    j.set("path", str(path));
    j.set("bytes", Json::integer((int64_t) content.size()));
    return j;
}

Json tool_edit_file(const Json & params, const std::string & cwd) {
    std::string path;
    std::string missing;
    if (!required_string(params, "path", path, missing)) {
        return error(missing);
    }
    const Json * edits = param(params, "edits");
    if (edits == nullptr || !edits->is_array() || edits->size() == 0) {
        return error("\"edits\" must be a non-empty array");
    }
    const fs::path p = resolve(cwd, path);
    std::string content;
    if (!read_file_bytes(p, content)) {
        return error("failed to open file: " + path);
    }
    // Match every edit against the *original* content, require one occurrence
    // each, and reject overlaps: an edit applied to a stale position is worse
    // than an error the model can retry.
    struct Match {
        size_t begin = 0;
        size_t end = 0;
        std::string replacement;
    };
    std::vector<Match> matches;
    for (size_t i = 0; i < edits->size(); i++) {
        const Json & e = edits->items()[i];
        if (!e.is_object() || !e.get("old_text").is_string() || !e.get("new_text").is_string()) {
            return error("every edit needs old_text and new_text strings");
        }
        const std::string old_text = e.get("old_text").as_string();
        const std::string new_text = e.get("new_text").as_string();
        if (old_text.empty()) {
            return error("edits[" + std::to_string(i) + "].old_text must not be empty");
        }
        const size_t at = content.find(old_text);
        if (at == std::string::npos) {
            return error("edits[" + std::to_string(i) + "].old_text not found in " + path);
        }
        if (content.find(old_text, at + 1) != std::string::npos) {
            return error("edits[" + std::to_string(i) + "].old_text is not unique in " + path);
        }
        matches.push_back({at, at + old_text.size(), new_text});
    }
    std::sort(matches.begin(), matches.end(), [](const Match & a, const Match & b) { return a.begin < b.begin; });
    for (size_t i = 1; i < matches.size(); i++) {
        if (matches[i].begin < matches[i - 1].end) {
            return error("edits overlap: they must be independent regions");
        }
    }
    std::string out;
    size_t at = 0;
    for (const Match & m : matches) {
        out.append(content, at, m.begin - at);
        out += m.replacement;
        at = m.end;
    }
    out.append(content, at, std::string::npos);
    if (out == content) {
        return error("no changes made: the replacement(s) produced identical content");
    }
    std::ofstream file(p, std::ios::binary | std::ios::trunc);
    if (!file) {
        return error("failed to write file: " + path);
    }
    file.write(out.data(), (std::streamsize) out.size());
    file.close();
    if (!file) {
        return error("failed to write file: " + path);
    }
    Json j = Json::object();
    j.set("result", str("file edited successfully"));
    j.set("path", str(path));
    j.set("edits_applied", Json::integer((int64_t) matches.size()));
    return j;
}

Json tool_file_glob_search(const Json & params, const std::string & cwd) {
    const std::string path = string_param(params, "path", ".");
    const std::string include = string_param(params, "include", "**");
    const std::string exclude = string_param(params, "exclude");
    const std::string type = string_param(params, "type", "file");
    if (!(type == "file" || type == "dir" || type == "all")) {
        return error("invalid type: " + type + " (expected \"file\", \"dir\" or \"all\")");
    }
    const int64_t limit_req = int_param(params, "limit", (int64_t) kListDefaultLimit);
    if (limit_req < 1) {
        return error("invalid limit: " + std::to_string(limit_req) + " (expected 1 or more)");
    }
    const size_t limit = std::min((size_t) limit_req, kListMaxLimit);
    const fs::path base = resolve(cwd, path);
    std::error_code ec;
    if (!fs::exists(base, ec)) {
        return error("path does not exist: " + path);
    }
    std::vector<LsEntry> all;
    bool truncated = false;
    list_entries(base, all, truncated);
    std::vector<LsEntry> matches;
    size_t total = 0;
    for (const LsEntry & e : all) {
        if (type == "file" && e.is_dir) continue;
        if (type == "dir" && !e.is_dir) continue;
        if (!e.is_dir && !glob_match(e.rel, include)) continue;  // include filters files
        if (!exclude.empty() && glob_match(e.rel, exclude)) continue;
        total++;
        if (matches.size() < limit) {
            matches.push_back(e);
        }
    }
    std::ostringstream text;
    Json entries = Json::array();
    for (const LsEntry & e : matches) {
        text << e.rel << (e.is_dir ? "/" : "") << "\n";
        Json entry = Json::object();
        entry.set("path", str(e.rel));
        entry.set("type", str(e.is_dir ? "dir" : "file"));
        entries.push(std::move(entry));
    }
    text << "\n---\nTotal matches: " << total << "\n";
    if (total > matches.size()) {
        text << "[" << matches.size() << " results limit reached (" << total
             << " total matches). Refine the glob pattern to narrow the search.]\n";
    }
    if (truncated) {
        text << "[results truncated: time budget or unreadable directory]\n";
    }
    Json j = Json::object();
    j.set("plain_text_response", str(text.str()));
    j.set("entries", std::move(entries));
    j.set("base", str(path_str(base)));
    return j;
}

Json tool_grep_search(const Json & params, const std::string & cwd) {
    std::string path;
    std::string missing;
    if (!required_string(params, "path", path, missing)) {
        return error(missing);
    }
    std::string pat_str;
    if (!required_string(params, "pattern", pat_str, missing)) {
        return error(missing);
    }
    const std::string include = string_param(params, "include", "**");
    const std::string exclude = string_param(params, "exclude");
    const bool show_lineno = bool_param(params, "return_line_numbers", false);
    const bool literal = bool_param(params, "literal", false);
    const bool ignore_case = bool_param(params, "ignore_case", false);
    const int64_t ctx_lines = std::max<int64_t>(0, int_param(params, "context_lines", 0));

    std::string pattern_src = pat_str;
    if (literal) {
        static const std::string kSpecials = "\\^$.|?*+()[]{}";
        std::string escaped;
        for (const char c : pat_str) {
            if (kSpecials.find(c) != std::string::npos) {
                escaped += '\\';
            }
            escaped += c;
        }
        pattern_src = escaped;
    }
    std::regex pattern;
    try {
        pattern = std::regex(pattern_src, ignore_case ? std::regex::ECMAScript | std::regex::icase
                                                      : std::regex::ECMAScript);
    } catch (const std::regex_error & e) {
        return error(std::string("invalid regex: ") + e.what());
    }

    const fs::path target = resolve(cwd, path);
    std::error_code ec;
    if (!fs::exists(target, ec)) {
        return error("path does not exist: " + path);
    }
    const bool single_file = fs::is_regular_file(target, ec);
    const fs::path base = single_file ? target.parent_path() : target;
    std::vector<std::string> files;
    if (single_file) {
        files.push_back(path_str(target));
    } else {
        std::vector<LsEntry> entries;
        bool truncated = false;
        list_entries(base, entries, truncated);
        for (const LsEntry & e : entries) {
            if (!e.is_dir && glob_match(e.rel, include) && (exclude.empty() || !glob_match(e.rel, exclude))) {
                files.push_back(path_str(base / e.rel));
            }
        }
        std::sort(files.begin(), files.end());
    }

    std::ostringstream text;
    size_t total = 0;
    bool limit_reached = false;
    for (const std::string & f : files) {
        if (total >= kSearchMaxResults) {
            limit_reached = true;
            break;
        }
        std::string content;
        if (!read_file_bytes(f, content) || content.find('\0') != std::string::npos) {
            continue;  // unreadable or binary
        }
        const std::vector<std::string> lines = split_lines(content);
        const std::string display = single_file ? path : rel_to(base, fs::path(f));
        for (size_t i = 0; i < lines.size() && total < kSearchMaxResults; i++) {
            if (!std::regex_search(lines[i], pattern)) {
                continue;
            }
            const int64_t from = ctx_lines > 0 ? std::max<int64_t>(0, (int64_t) i - ctx_lines) : (int64_t) i;
            const int64_t to = ctx_lines > 0 ? std::min<int64_t>((int64_t) lines.size() - 1, (int64_t) i + ctx_lines)
                                             : (int64_t) i;
            for (int64_t j = from; j <= to; j++) {
                const bool is_match = j == (int64_t) i;
                text << display << (is_match ? ':' : '-');
                if (show_lineno) {
                    text << (j + 1) << (is_match ? ':' : '-');
                }
                text << lines[(size_t) j] << "\n";
            }
            if (ctx_lines > 0) {
                text << "--\n";
            }
            total++;
        }
    }
    text << "\n---\nTotal matches: " << total << "\n";
    if (limit_reached) {
        text << "[" << kSearchMaxResults << " matches limit reached. Narrow the path/pattern/include to see more.]\n";
    }
    return plain(text.str());
}

Json tool_exec_shell_command(const Json & params, const std::string & cwd) {
    std::string command;
    std::string missing;
    if (!required_string(params, "command", command, missing)) {
        return error(missing);
    }
    int64_t timeout = int_param(params, "timeout", kExecDefaultTimeout);
    timeout = std::max<int64_t>(1, std::min<int64_t>(timeout, kExecMaxTimeout));
    size_t max_output = (size_t) int_param(params, "max_output_size", (int64_t) kExecMaxOutput);
    max_output = std::max<size_t>(1, std::min(max_output, kExecMaxOutput));
    const ProcResult r = run_process({"/bin/sh", "-c", command}, path_str(base_dir(cwd)), (int) timeout, max_output);
    if (r.spawn_error) {
        return error("failed to run the command");
    }
    std::string text = r.output;
    text += "\n[exit code: " + std::to_string(r.exit_code) + "]";
    if (r.timed_out) {
        text += " [exit due to timed out]";
    }
    return plain(text);
}

Json tool_get_info(const Json & params, const std::string & cwd) {
    std::string os_info = "unknown";
    utsname u{};
    if (uname(&u) == 0) {
        os_info = std::string(u.sysname) + " " + u.release + " " + u.version + " " + u.machine;
    }
    std::string dir = string_param(params, "cwd");
    if (dir.empty()) {
        dir = path_str(base_dir(cwd));
    }
    Json j = Json::object();
    j.set("os", str(os_info));
    j.set("cwd", str(dir));
    return j;
}

//
// the registries
//

struct ToolDef {
    const char * name;
    const char * display;
    bool write;
    const char * description;
    const char * parameters;  // the JSON schema, parsed by list()
};

const ToolDef kTools[] = {
    {"read_file", "Read file", false,
     "Read the contents of a file. Optionally specify a 1-based line range. "
     "If append_loc is true, each line is prefixed with its line number (e.g. \"1\u2192...\").",
     R"json({"type":"object","properties":{
        "path":{"type":"string","description":"Path to the file"},
        "start_line":{"type":"integer","description":"First line to read, 1-based (default: 1)"},
        "end_line":{"type":"integer","description":"Last line to read, 1-based inclusive (default: end of file)"},
        "append_loc":{"type":"boolean","description":"Prefix each line with its line number"}},
        "required":["path"]})json"},
    {"file_glob_search", "File search", false,
     "Recursively search for files matching a glob pattern under a directory. "
     "Automatically skips files ignored by .gitignore (when the directory is inside a git repo) "
     "and common junk directories (.git, node_modules, build, dist, etc.) otherwise. "
     "A pattern with no '/' (e.g. \"*.cpp\") matches the file's basename at any depth. "
     "A pattern containing '/' matches the full relative path; unless already anchored with "
     "\"**/\" or a leading '/', it is automatically prefixed with \"**/\". "
     "Use type=\"dir\" or \"all\" to also list directories; directory entries are suffixed with '/' in the output. "
     "Note: directory listings do not apply .gitignore filtering.",
     R"json({"type":"object","properties":{
        "path":{"type":"string","description":"Base directory to search in"},
        "include":{"type":"string","description":"Glob pattern for files to include (e.g. \"*.cpp\" or \"src/**/*.cpp\"). Default: **"},
        "exclude":{"type":"string","description":"Glob pattern to exclude files"},
        "type":{"type":"string","description":"\"file\" (default), \"dir\" or \"all\""},
        "limit":{"type":"integer","description":"Maximum number of results to return, capped at 100 (default 100)"}},
        "required":["path"]})json"},
    {"grep_search", "Grep search", false,
     "Search for a pattern in files under a path. Returns matching lines with file paths "
     "(and, unless searching a single file, paths relative to the given directory). "
     "Automatically skips files ignored by .gitignore (when the directory is inside a git repo) "
     "and common junk directories (.git, node_modules, build, dist, etc.) otherwise. "
     "include/exclude: a pattern with no '/' matches the basename at any depth; a pattern "
     "containing '/' matches the full relative path (auto-anchored with \"**/\" unless already anchored).",
     R"json({"type":"object","properties":{
        "path":{"type":"string","description":"File or directory to search in"},
        "pattern":{"type":"string","description":"Pattern to search for (regular expression unless literal is true)"},
        "include":{"type":"string","description":"Glob pattern to filter files (default: **)"},
        "exclude":{"type":"string","description":"Glob pattern to exclude files"},
        "return_line_numbers":{"type":"boolean","description":"If true, include line numbers in results"},
        "literal":{"type":"boolean","description":"Treat pattern as a literal string instead of a regular expression (default: false)"},
        "ignore_case":{"type":"boolean","description":"Case-insensitive search (default: false)"},
        "context_lines":{"type":"integer","description":"Number of lines of context to show before and after each match (default: 0)"}},
        "required":["path","pattern"]})json"},
    {"exec_shell_command", "Execute shell command", true,
     "Execute a shell command and return its output (stdout and stderr combined).",
     R"json({"type":"object","properties":{
        "command":{"type":"string","description":"Shell command to execute"},
        "timeout":{"type":"integer","description":"Timeout in seconds (default 10, max 60)"},
        "max_output_size":{"type":"integer","description":"Maximum output size in bytes (default 16384)"}},
        "required":["command"]})json"},
    {"write_file", "Write file", true,
     "Write content to a file, creating it (including parent directories) if it does not exist. "
     "May use with edit_file for more complex edits.",
     R"json({"type":"object","properties":{
        "path":{"type":"string","description":"Path of the file to write"},
        "content":{"type":"string","description":"Content to write"}},
        "required":["path","content"]})json"},
    {"edit_file", "Edit file", true,
     "Edit a file using exact text replacement. Each edits[].old_text must be unique in the file "
     "and is matched against the original content, not incrementally. Merge nearby changes into "
     "one edit instead of overlapping edits. Use write_file to replace the whole file.",
     R"json({"type":"object","properties":{
        "path":{"type":"string","description":"Path to the file to edit"},
        "edits":{"type":"array","description":"One or more exact text replacements to apply","items":{
            "type":"object","properties":{
                "old_text":{"type":"string","description":"Exact text to find; must be unique in the file and must not overlap with other edits"},
                "new_text":{"type":"string","description":"Text to replace old_text with"}},
            "required":["old_text","new_text"]}}},
        "required":["path","edits"]})json"},
    {"get_info", "Get Runtime Info", false,
     "Returns runtime info: the OS name/version and the current working directory",
     R"json({"type":"object","properties":{}})json"},
};

const ToolDef * find_def(const std::string & name) {
    for (const ToolDef & t : kTools) {
        if (name == t.name) {
            return &t;
        }
    }
    return nullptr;
}

}  // namespace

std::vector<std::string> available() {
    std::vector<std::string> names;
    for (const ToolDef & t : kTools) {
        names.emplace_back(t.name);
    }
    return names;
}

std::vector<std::string> parse(const std::string & spec) {
    std::vector<std::string> enabled;
    if (spec.empty()) {
        return enabled;
    }
    size_t pos = 0;
    while (pos <= spec.size()) {
        const size_t comma = spec.find(',', pos);
        std::string name = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        while (!name.empty() && (name.front() == ' ' || name.front() == '\t')) name.erase(0, 1);
        while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) name.pop_back();
        if (!name.empty()) {
            if (name == "all") {
                return available();
            }
            if (find_def(name) == nullptr) {
                std::string names;
                for (const std::string & n : available()) {
                    names += (names.empty() ? "" : ", ") + n;
                }
                throw std::runtime_error("unknown tool \"" + name + "\". available tools: " + names);
            }
            if (std::find(enabled.begin(), enabled.end(), name) == enabled.end()) {
                enabled.push_back(name);
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return enabled;
}

Json list(const std::vector<std::string> & enabled) {
    Json arr = Json::array();
    for (const std::string & name : enabled) {
        const ToolDef * def = find_def(name);
        if (def == nullptr) {
            continue;
        }
        Json definition = Json::object();
        definition.set("type", str("function"));
        Json function = Json::object();
        function.set("name", str(def->name));
        function.set("description", str(def->description));
        function.set("parameters", Json::parse(def->parameters));
        definition.set("function", std::move(function));
        Json permissions = Json::object();
        permissions.set("write", boolean(def->write));
        Json j = Json::object();
        j.set("display_name", str(def->display));
        j.set("tool", str(def->name));
        j.set("type", str("server"));
        j.set("permissions", std::move(permissions));
        j.set("uses_cwd", boolean(true));
        j.set("definition", std::move(definition));
        arr.push(std::move(j));
    }
    return arr;
}

Json invoke(const std::string & name, const Json & params, const std::string & cwd) {
    if (name == "read_file") return tool_read_file(params, cwd);
    if (name == "write_file") return tool_write_file(params, cwd);
    if (name == "edit_file") return tool_edit_file(params, cwd);
    if (name == "file_glob_search") return tool_file_glob_search(params, cwd);
    if (name == "grep_search") return tool_grep_search(params, cwd);
    if (name == "exec_shell_command") return tool_exec_shell_command(params, cwd);
    if (name == "get_info") return tool_get_info(params, cwd);
    throw std::runtime_error("unknown tool \"" + name + "\"");
}

}  // namespace omph::server::tools
