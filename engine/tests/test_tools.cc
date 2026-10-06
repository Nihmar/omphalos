// The server-side agent tools (#380), on a temporary tree: the registry, the
// JSON shapes the web UI reads, and every tool's success and failure paths.
#include "check.hh"
#include "server/tools.hh"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using omph::text::Json;
using namespace omph::server;

// A temporary directory, removed when the test ends.
struct Sandbox {
    fs::path dir;
    Sandbox() {
        dir = fs::temp_directory_path() / ("omph-tools-test-" + std::to_string(::getpid()));
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    ~Sandbox() { fs::remove_all(dir); }
    void write(const std::string & rel, const std::string & content) const {
        fs::create_directories((dir / rel).parent_path());
        std::ofstream f(dir / rel, std::ios::binary | std::ios::trunc);
        f.write(content.data(), (std::streamsize) content.size());
    }
    std::string read(const std::string & rel) const {
        std::ifstream f(dir / rel, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
};

Json run(const std::string & tool, const Json & params, const std::string & cwd) {
    return tools::invoke(tool, params, cwd);
}

Json params_of(const std::string & json) { return Json::parse(json); }

std::string text_of(const Json & r) { return r.get("plain_text_response").as_string(); }
bool has_error(const Json & r) { return r.has("error") && r.get("error").is_string(); }

void test_parse() {
    CHECK(tools::parse("").empty(), "no spec: no tools");
    CHECK(tools::parse("all").size() == 7, "all: the seven");
    const std::vector<std::string> two = tools::parse("read_file,grep_search");
    CHECK(two.size() == 2 && two[0] == "read_file" && two[1] == "grep_search", "a list keeps its order");
    CHECK(tools::parse("read_file, read_file").size() == 1, "a duplicate is one tool");
    bool threw = false;
    std::string message;
    try {
        tools::parse("read_file,nope");
    } catch (const std::runtime_error & e) {
        threw = true;
        message = e.what();
    }
    CHECK(threw, "an unknown name throws");
    CHECK(message.find("nope") != std::string::npos && message.find("read_file") != std::string::npos,
          "the message names the tool and the available ones: %s", message.c_str());
}

void test_list() {
    const Json all = tools::list(tools::parse("all"));
    CHECK(all.is_array() && all.size() == 7, "the listing is the seven");
    const Json & first = all.items()[0];
    CHECK(first.get("tool").as_string() == "read_file", "the first is read_file");
    CHECK(first.get("type").as_string() == "server", "the type the UI filters on");
    CHECK(first.get("display_name").as_string() == "Read file", "the display name");
    CHECK(first.get("uses_cwd").is_bool() && first.get("uses_cwd").as_bool(), "the tools use a cwd");
    CHECK(first.get("permissions").get("write").is_bool() && !first.get("permissions").get("write").as_bool(),
          "read_file is read-only");
    const Json & fn = first.get("definition").get("function");
    CHECK(first.get("definition").get("type").as_string() == "function", "an OpenAI function");
    CHECK(fn.get("name").as_string() == "read_file" && fn.get("parameters").get("required").is_array(),
          "name and parameters schema");
    const Json & write = all.items()[4];
    CHECK(write.get("tool").as_string() == "write_file" && write.get("permissions").get("write").as_bool(),
          "write_file asks for write permission");
    const Json only = tools::list({"get_info"});
    CHECK(only.size() == 1 && only.items()[0].get("tool").as_string() == "get_info", "a subset lists one");
    CHECK(tools::list({}).size() == 0, "nothing enabled lists nothing");
    try {
        tools::invoke("nope", Json::object(), "");
        CHECK(false, "an unknown invocation throws");
    } catch (const std::runtime_error &) {
        CHECK(true, "an unknown invocation throws");
    }
}

void test_read_write(const Sandbox & s) {
    const Json w = run("write_file", params_of(R"({"path":"d/a.txt","content":"one\ntwo\nthree\n"})"),
                       s.dir.generic_string());
    CHECK(w.get("result").as_string() == "file written successfully" && w.get("bytes").as_number() == 14,
          "write_file creates the parent and reports the bytes");
    CHECK(s.read("d/a.txt") == "one\ntwo\nthree\n", "the bytes are on disk");

    CHECK(text_of(run("read_file", params_of(R"({"path":"d/a.txt"})"), s.dir.generic_string())) ==
              "one\ntwo\nthree\n",
          "read_file returns the file");
    CHECK(text_of(run("read_file", params_of(R"({"path":"d/a.txt","start_line":2,"end_line":2})"),
                      s.dir.generic_string())) == "two\n",
          "a 1-based line range");
    CHECK(text_of(run("read_file", params_of(R"({"path":"d/a.txt","append_loc":true})"),
                      s.dir.generic_string())) == "1\u2192one\n2\u2192two\n3\u2192three\n",
          "append_loc prefixes the numbers");
    CHECK(has_error(run("read_file", params_of(R"({"path":"missing"})"), s.dir.generic_string())),
          "a missing file is an error");
    CHECK(has_error(run("read_file", params_of(R"({"start_line":1})"), s.dir.generic_string())),
          "a missing path is an error");

    const std::string big(17 * 1024, 'x');
    run("write_file", params_of(R"({"path":"big.txt","content":""})"), s.dir.generic_string());
    s.write("big.txt", big);
    const Json r = run("read_file", params_of(R"({"path":"big.txt"})"), s.dir.generic_string());
    CHECK(has_error(r) && r.get("error").as_string().find("start_line") != std::string::npos,
          "a file over 16 KB needs a range: %s", has_error(r) ? r.get("error").as_string().c_str() : "?");
    CHECK(text_of(run("read_file", params_of(R"({"path":"big.txt","start_line":1,"end_line":2})"),
                      s.dir.generic_string()))
              .size() > 0,
          "and a range reads it");
}

void test_edit(const Sandbox & s) {
    s.write("e.txt", "alpha\nbeta gamma\ndelta\n");
    const std::string cwd = s.dir.generic_string();
    Json r = run("edit_file", params_of(R"({"path":"e.txt","edits":[{"old_text":"beta","new_text":"BETA"}]})"), cwd);
    CHECK(r.get("result").as_string() == "file edited successfully" && r.get("edits_applied").as_number() == 1,
          "one edit");
    CHECK(s.read("e.txt") == "alpha\nBETA gamma\ndelta\n", "the edit is on disk");

    s.write("e.txt", "aa\naa\n");
    r = run("edit_file", params_of(R"({"path":"e.txt","edits":[{"old_text":"aa","new_text":"bb"}]})"), cwd);
    CHECK(has_error(r) && r.get("error").as_string().find("not unique") != std::string::npos,
          "a non-unique old_text is refused");

    s.write("e.txt", "abcdef\n");
    r = run("edit_file", params_of(R"({"path":"e.txt","edits":[{"old_text":"abc","new_text":"x"},
         {"old_text":"cde","new_text":"y"}]})"),
            cwd);
    CHECK(has_error(r) && r.get("error").as_string().find("overlap") != std::string::npos,
          "overlapping edits are refused");

    r = run("edit_file", params_of(R"({"path":"e.txt","edits":[{"old_text":"zzz","new_text":"y"}]})"), cwd);
    CHECK(has_error(r) && r.get("error").as_string().find("not found") != std::string::npos, "not found is an error");

    r = run("edit_file", params_of(R"({"path":"e.txt","edits":[{"old_text":"abc","new_text":"abc"}]})"), cwd);
    CHECK(has_error(r) && r.get("error").as_string().find("no changes") != std::string::npos, "a no-op edit is refused");

    r = run("edit_file", params_of(R"({"path":"e.txt","edits":[]})"), cwd);
    CHECK(has_error(r) && r.get("error").as_string().find("non-empty") != std::string::npos, "no edits is an error");

    r = run("edit_file", params_of(R"({"path":"missing.txt","edits":[{"old_text":"a","new_text":"b"}]})"), cwd);
    CHECK(has_error(r), "editing a missing file is an error");
}

void test_grep(const Sandbox & s) {
    s.write("g/one.txt", "Alpha\nbeta\ngamma\n");
    s.write("g/sub/two.txt", "beta again\n");
    s.write("g/three.md", "beta in markdown\n");
    const std::string cwd = s.dir.generic_string();
    Json r = run("grep_search", params_of(R"({"path":"g","pattern":"beta"})"), cwd);
    CHECK(!has_error(r) && text_of(r).find("one.txt:beta") != std::string::npos &&
              text_of(r).find("three.md:beta") != std::string::npos && text_of(r).find("Total matches: 3") != std::string::npos,
          "grep finds every file: %s", text_of(r).c_str());
    r = run("grep_search", params_of(R"({"path":"g","pattern":"beta","include":"*.md"})"), cwd);
    CHECK(text_of(r).find("three.md") != std::string::npos && text_of(r).find("one.txt") == std::string::npos,
          "include filters files");
    r = run("grep_search", params_of(R"({"path":"g","pattern":"BETA","ignore_case":true,"return_line_numbers":true})"),
            cwd);
    CHECK(text_of(r).find("beta") != std::string::npos && text_of(r).find(":2:") != std::string::npos,
          "ignore_case and line numbers");
    r = run("grep_search", params_of(R"({"path":"g","pattern":"b.t","literal":true})"), cwd);
    CHECK(text_of(r).find("Total matches: 0") != std::string::npos, "literal treats the dot as a dot");
    r = run("grep_search", params_of(R"({"path":"g/one.txt","pattern":"gamma","return_line_numbers":true})"), cwd);
    CHECK(text_of(r).find("g/one.txt:3:gamma") != std::string::npos, "a single file keeps its path");
    r = run("grep_search", params_of(R"({"path":"g","pattern":"beta","context_lines":1})"), cwd);
    CHECK(text_of(r).find("Alpha") != std::string::npos && text_of(r).find("--") != std::string::npos,
          "context lines around a match");
    r = run("grep_search", params_of(R"({"path":"g","pattern":"(","literal":false})"), cwd);
    CHECK(has_error(r), "an invalid regex is an error");
    r = run("grep_search", params_of(R"({"path":"nope","pattern":"x"})"), cwd);
    CHECK(has_error(r), "a missing path is an error");
}

void test_glob(const Sandbox & s) {
    s.write("f/a.txt", "a\n");
    s.write("f/b.md", "b\n");
    s.write("f/sub/c.txt", "c\n");
    const std::string cwd = s.dir.generic_string();
    Json r = run("file_glob_search", params_of(R"({"path":"f","include":"*.txt"})"), cwd);
    CHECK(!has_error(r) && r.get("entries").size() == 2, "a basename pattern matches at any depth: %s",
          text_of(r).c_str());
    CHECK(r.get("base").as_string().find("/f") != std::string::npos, "base is absolute");
    r = run("file_glob_search", params_of(R"({"path":"f","include":"sub/*.txt"})"), cwd);
    CHECK(r.get("entries").size() == 1 && r.get("entries").items()[0].get("path").as_string() == "sub/c.txt",
          "a pattern with a slash is anchored");
    r = run("file_glob_search", params_of(R"({"path":"f","include":"*","type":"dir"})"), cwd);
    CHECK(r.get("entries").size() == 1 && r.get("entries").items()[0].get("path").as_string() == "sub",
          "directories only");
    r = run("file_glob_search", params_of(R"({"path":"f","include":"*","limit":1})"), cwd);
    CHECK(r.get("entries").size() == 1 && text_of(r).find("limit reached") != std::string::npos, "the limit is applied");
    r = run("file_glob_search", params_of(R"({"path":"f","include":"*","exclude":"sub/*"})"), cwd);
    CHECK(r.get("entries").size() == 2, "exclude drops a subtree");
    r = run("file_glob_search", params_of(R"({"path":"nope"})"), cwd);
    CHECK(has_error(r), "a missing path is an error");
}

void test_exec(const Sandbox & s) {
    const std::string cwd = s.dir.generic_string();
    Json r = run("exec_shell_command", params_of(R"({"command":"echo hi; pwd"})"), cwd);
    CHECK(!has_error(r) && text_of(r).find("hi\n") == 0 && text_of(r).find(s.dir.generic_string()) != std::string::npos &&
              text_of(r).find("[exit code: 0]") != std::string::npos,
          "the output and the exit code: %s", text_of(r).c_str());
    r = run("exec_shell_command", params_of(R"({"command":"exit 3"})"), cwd);
    CHECK(text_of(r).find("[exit code: 3]") != std::string::npos, "a failing command keeps its code");
    r = run("exec_shell_command", params_of(R"({"command":"sleep 5","timeout":1})"), cwd);
    CHECK(text_of(r).find("[exit due to timed out]") != std::string::npos, "a timeout is reported");
    r = run("exec_shell_command", params_of(R"({})"), cwd);
    CHECK(has_error(r), "a missing command is an error");
}

void test_get_info(const Sandbox & s) {
    const Json r = run("get_info", Json::object(), s.dir.generic_string());
    CHECK(!has_error(r) && r.get("os").as_string().find("Linux") != std::string::npos, "the OS");
    CHECK(r.get("cwd").as_string() == s.dir.generic_string(), "the cwd override");
    const Json by_param = run("get_info", params_of(R"({"cwd":"/tmp"})"), s.dir.generic_string());
    CHECK(by_param.get("cwd").as_string() == "/tmp", "params.cwd wins");
}

}  // namespace

int main() {
    test_parse();
    test_list();
    const Sandbox s;
    test_read_write(s);
    test_edit(s);
    test_grep(s);
    test_glob(s);
    test_exec(s);
    test_get_info(s);
    if (omph_test::failures == 0) {
        std::printf("test_tools: ok\n");
    }
    return omph_test::failures;
}
