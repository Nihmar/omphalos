#include "server/http.hh"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace omph::server {
namespace {

constexpr size_t kMaxHead = 64 << 10;
constexpr size_t kMaxBody = 64 << 20;
// A client that stalls must not hold the one-connection server (#338): one
// recv() may block this long, and reading a whole request (headers + body)
// has a deadline of its own, so a byte-every-few-seconds client cannot keep
// the server forever.
constexpr int kRecvTimeoutMs = 10'000;
constexpr int kRequestDeadlineMs = 120'000;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

const char * reason(const int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 415: return "Unsupported Media Type";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "Unknown";
    }
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

std::string strip(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) --e;
    return s.substr(b, e - b);
}

std::string error_json(const std::string & message) {
    std::string m;
    for (const char c : message) {
        if (c == '"' || c == '\\') m += '\\';
        m += c;
    }
    return "{\"error\": {\"message\": \"" + m + "\", \"type\": \"invalid_request_error\", \"param\": null, "
           "\"code\": null}}";
}

} // namespace

const std::string * Request::header(const std::string_view name) const {
    for (const auto & [k, v] : headers) {
        if (k == name) return &v;
    }
    return nullptr;
}

Connection::~Connection() {
    if (fd_ >= 0) {
        ::shutdown(fd_, SHUT_WR);
        ::close(fd_);
    }
}

bool Connection::write_all(const std::string & bytes) {
    size_t off = 0;
    while (off < bytes.size()) {
        const ssize_t n = ::send(fd_, bytes.data() + off, bytes.size() - off, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        off += (size_t) n;
    }
    return true;
}

std::string Connection::head(const int status, const std::string & content_type, const int64_t length) const {
    std::string h = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\n";
    if (!content_type.empty()) h += "Content-Type: " + content_type + "\r\n";
    if (length >= 0) h += "Content-Length: " + std::to_string(length) + "\r\n";
    for (const auto & [k, v] : extra_headers) h += k + ": " + v + "\r\n";
    h += "Connection: close\r\n\r\n";
    return h;
}

bool Connection::respond(const int status, const std::string & content_type, const std::string & body) {
    return write_all(head(status, content_type, (int64_t) body.size()) + body);
}

bool Connection::begin_events() {
    std::string h = head(200, "text/event-stream", -1);
    h.insert(h.size() - 2, "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n");
    streaming_ = write_all(h);
    return streaming_;
}

bool Connection::event(const std::string & data) { return write_all("data: " + data + "\n\n"); }

bool Connection::client_gone() {
    char peek = 0;
    const ssize_t n = ::recv(fd_, &peek, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n > 0) {
        return false;  // nothing says the client stopped reading
    }
    if (n == 0) {
        return true;   // orderly shutdown (or a half-close: it is not waiting for us)
    }
    return errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR;
}

// #364: an early error's response is written while the client may still be
// sending; reading (and dropping) that data keeps close() from turning into an
// RST, which can make the client lose the response.
void Connection::drain(const size_t limit) {
    timeval tv{1, 0};
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    size_t seen = 0;
    char buf[16384];
    while (seen < limit) {
        const ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        seen += (size_t) n;
    }
}

bool Connection::read(Request & req) {
    // a client that stalls must not hold the server (one connection at a time)
    timeval tv{kRecvTimeoutMs / 1000, (kRecvTimeoutMs % 1000) * 1000};
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    const int one = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    const int64_t deadline = now_ms() + kRequestDeadlineMs;
    std::string data;
    char buf[16384];
    size_t end = std::string::npos;
    while ((end = data.find("\r\n\r\n")) == std::string::npos) {
        if (data.size() > kMaxHead) {
            respond(431, "application/json", error_json("request headers too large"));
            drain();  // the client may still be sending them (#364)
            return false;
        }
        if (now_ms() > deadline) {
            respond(408, "application/json", error_json("the request timed out"));
            return false;
        }
        const ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            if (!data.empty()) respond(408, "application/json", error_json("incomplete request"));
            return false;
        }
        data.append(buf, (size_t) n);
    }
    // request line, headers
    size_t pos = data.find("\r\n");
    const std::string line = data.substr(0, pos);
    const size_t s1 = line.find(' '), s2 = line.rfind(' ');
    if (s1 == std::string::npos || s2 == s1 || line.compare(s2 + 1, 5, "HTTP/") != 0) {
        respond(400, "application/json", error_json("malformed request line"));
        return false;
    }
    req.method = line.substr(0, s1);
    req.path = line.substr(s1 + 1, s2 - s1 - 1);
    if (const size_t q = req.path.find('?'); q != std::string::npos) req.path.resize(q);
    while (pos < end) {
        const size_t next = data.find("\r\n", pos + 2);
        const std::string h = data.substr(pos + 2, next - pos - 2);
        pos = next;
        if (h.empty()) continue;
        const size_t colon = h.find(':');
        if (colon == std::string::npos) {
            respond(400, "application/json", error_json("malformed header"));
            return false;
        }
        req.headers.emplace_back(lower(strip(h.substr(0, colon))), strip(h.substr(colon + 1)));
    }
    if (const std::string * te = req.header("transfer-encoding"); te != nullptr && lower(*te) != "identity") {
        respond(501, "application/json", error_json("chunked request bodies are not supported"));
        return false;
    }
    size_t length = 0;
    if (const std::string * cl = req.header("content-length")) {
        char * e = nullptr;
        const unsigned long long v = std::strtoull(cl->c_str(), &e, 10);
        if (cl->empty() || *e != '\0') {
            respond(400, "application/json", error_json("bad Content-Length"));
            return false;
        }
        if (v > kMaxBody) {
            respond(413, "application/json", error_json("request body too large"));
            drain();  // the body is still in flight: do not reset before the client reads (#364)
            return false;
        }
        length = (size_t) v;
    } else if (req.method == "POST") {
        respond(411, "application/json", error_json("Content-Length required"));
        return false;
    }
    if (const std::string * ex = req.header("expect"); ex != nullptr && lower(*ex) == "100-continue") {
        if (!write_all("HTTP/1.1 100 Continue\r\n\r\n")) return false;
    }
    req.body = data.substr(end + 4);
    while (req.body.size() < length) {
        if (now_ms() > deadline) {
            respond(408, "application/json", error_json("the request timed out"));
            return false;
        }
        const ssize_t n = ::recv(fd_, buf, std::min(sizeof(buf), length - req.body.size()), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            respond(408, "application/json", error_json("incomplete request body"));
            return false;
        }
        req.body.append(buf, (size_t) n);
    }
    req.body.resize(length);
    return true;
}

Listener::Listener(const std::string & host, const int port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
    addrinfo * res = nullptr;
    const std::string service = std::to_string(port);
    if (const int rc = ::getaddrinfo(host.c_str(), service.c_str(), &hints, &res); rc != 0) {
        throw std::runtime_error("cannot resolve " + host + ": " + gai_strerror(rc));
    }
    std::string err = "no address";
    for (addrinfo * a = res; a != nullptr; a = a->ai_next) {
        const int fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) {
            err = std::strerror(errno);
            continue;
        }
        const int one = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (::bind(fd, a->ai_addr, a->ai_addrlen) == 0 && ::listen(fd, 16) == 0) {
            fd_ = fd;
            break;
        }
        err = std::strerror(errno);
        ::close(fd);
    }
    ::freeaddrinfo(res);
    if (fd_ < 0) {
        throw std::runtime_error("cannot listen on " + host + ":" + service + ": " + err);
    }
}

Listener::~Listener() {
    if (fd_ >= 0) ::close(fd_);
}

int Listener::accept_one() {
    while (true) {
        const int fd = ::accept(fd_, nullptr, nullptr);
        if (fd >= 0 || errno != EINTR) return fd;
    }
}

} // namespace omph::server
