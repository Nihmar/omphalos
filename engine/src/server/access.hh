// Who may drive the server, and with what: the checks the HTTP layer makes
// before a request reaches the engine (#338). HIP-free on purpose: the CPU
// tests drive them like they drive the OpenAI parser.
#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace omph::server {

inline std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

// RFC 7235: the auth scheme is case-insensitive; the key comparison is
// constant-time, so a wrong key cannot be guessed one byte at a time.
inline bool bearer_key_ok(const std::string & authorization, const std::string & key) {
    static constexpr char kScheme[] = "bearer ";
    constexpr size_t k = sizeof(kScheme) - 1;
    if (authorization.size() != k + key.size()) {
        return false;
    }
    for (size_t i = 0; i < k; ++i) {
        if (std::tolower((unsigned char) authorization[i]) != kScheme[i]) {
            return false;
        }
    }
    unsigned char diff = 0;
    for (size_t i = 0; i < key.size(); ++i) {
        diff |= (unsigned char) authorization[k + i] ^ (unsigned char) key[i];
    }
    return diff == 0;
}

// The Host a client must present: a DNS-rebinding page reaches 127.0.0.1 with
// its own name in Host, so only the bound address and the loopback names pass.
// A missing Host (HTTP/1.0) is the caller's decision. A wildcard bind
// (0.0.0.0 / ::) cannot name itself and is taken as "anything": the check is
// for the default loopback bind, where the attack lives.
inline bool host_allowed(const std::string & host_header, const std::string & bind_host) {
    const std::string bind = lower_ascii(bind_host);
    if (bind.empty() || bind == "0.0.0.0" || bind == "::" || bind == "*") {
        return true;
    }
    std::string h = lower_ascii(host_header);
    if (!h.empty() && h[0] == '[') {  // a bracketed IPv6 literal keeps its brackets
        const size_t close = h.find(']');
        if (close != std::string::npos) h.resize(close + 1);
    } else if (const size_t first = h.find(':'); first != std::string::npos && first == h.rfind(':')) {
        h.resize(first);  // host:port; a bare IPv6 literal has more than one colon
    }
    if (h == bind) {
        return true;
    }
    for (const char * loopback : {"localhost", "127.0.0.1", "::1", "[::1]"}) {
        if (h == loopback) return true;
    }
    return false;
}

// A POST body must say it is JSON: a text/plain POST from a web page is a
// CORS-simple request, and it would otherwise start a generation.
inline bool json_content_type(const std::string & value) {
    const std::string v = lower_ascii(value);
    constexpr size_t n = sizeof("application/json") - 1;
    return v.compare(0, n, "application/json") == 0 &&
           (v.size() == n || v[n] == ';' || v[n] == ' ' || v[n] == '\t');
}

// Whether a --host names only the loopback interface: localhost and the
// 127.0.0.0/8 / ::1 literals. Anything else (a LAN address, a wildcard bind)
// is reachable from the network, so the server-side agent tools may not be
// enabled there without an API key (#387).
inline bool is_loopback_host(const std::string & host) {
    const std::string h = lower_ascii(host);
    if (h == "localhost" || h == "::1" || h == "[::1]") return true;
    unsigned parts[4] = {0, 0, 0, 0};
    int part = 0;
    for (const char c : h) {
        if (std::isdigit((unsigned char) c)) {
            parts[part] = parts[part] * 10 + (unsigned) (c - '0');
            if (parts[part] > 255) return false;
        } else if (c == '.' && part < 3) {
            ++part;
        } else {
            return false;
        }
    }
    return part == 3 && parts[0] == 127;
}

} // namespace omph::server
