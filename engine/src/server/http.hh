// A minimal HTTP/1.1 server for omph-server (#156): POSIX sockets, one
// connection at a time (one GPU, one sequence: requests are served in turn),
// Content-Length bodies, server-sent events, Connection: close. No dependency.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace omph::server {

struct Request {
    std::string method;
    std::string path;     // without the query string
    std::vector<std::pair<std::string, std::string>> headers;  // names in lower case
    std::string body;
    const std::string * header(std::string_view name) const;
};

// One accepted connection; closes on destruction.
class Connection {
public:
    explicit Connection(int fd) : fd_(fd) {}
    ~Connection();
    Connection(const Connection &) = delete;
    Connection & operator=(const Connection &) = delete;

    // Reads one request. On a malformed one it answers the error itself and
    // returns false.
    bool read(Request & req);
    // A whole response. Every write returns false once the client is gone.
    bool respond(int status, const std::string & content_type, const std::string & body);
    // A server-sent event stream: the headers, then one `data:` event per call.
    bool begin_events();
    bool event(const std::string & data);

    // True once the client closed (or reset) the connection: the request in
    // flight has nowhere to write its answer and must stop (#338). A poll of a
    // socket with nothing to read.
    bool client_gone();

    // added to every response (CORS)
    std::vector<std::pair<std::string, std::string>> extra_headers;

private:
    bool write_all(const std::string & bytes);
    std::string head(int status, const std::string & content_type, int64_t length) const;
    int fd_;
};

class Listener {
public:
    // Binds and listens; throws std::runtime_error.
    Listener(const std::string & host, int port);
    ~Listener();
    Listener(const Listener &) = delete;
    Listener & operator=(const Listener &) = delete;
    // Waits for the next connection: its descriptor, or -1 on an error.
    int accept_one();

private:
    int fd_ = -1;
};

} // namespace omph::server
