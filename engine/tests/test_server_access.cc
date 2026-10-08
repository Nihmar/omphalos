// The HTTP access checks (#338), without a socket: the auth scheme and key
// comparison, the Host allow list and the Content-Type rule.
#include "check.hh"
#include "server/access.hh"

#include <string>

namespace {

using omph::server::bearer_key_ok;
using omph::server::host_allowed;
using omph::server::is_loopback_host;
using omph::server::json_content_type;

void test_bearer() {
    CHECK(bearer_key_ok("Bearer secret", "secret"), "the exact key");
    CHECK(bearer_key_ok("bearer secret", "secret"), "the scheme is case-insensitive (RFC 7235)");
    CHECK(bearer_key_ok("BEARER secret", "secret"), "the scheme in capitals");
    CHECK(!bearer_key_ok("Bearer secre", "secret"), "a shorter key");
    CHECK(!bearer_key_ok("Bearer secrets", "secret"), "a longer key");
    CHECK(!bearer_key_ok("Basic secret", "secret"), "another scheme");
    CHECK(!bearer_key_ok("", "secret"), "no header");
    CHECK(!bearer_key_ok("Bearer ", "secret"), "no key");
    const std::string with_nul("s\0cret", 5);
    CHECK(bearer_key_ok("Bearer " + with_nul, with_nul), "a NUL in the key is still compared");
    CHECK(!bearer_key_ok("Bearer s\0CRET", with_nul), "and it is compared");
}

void test_host() {
    CHECK(host_allowed("127.0.0.1:8080", "127.0.0.1"), "the bound address with a port");
    CHECK(host_allowed("127.0.0.1", "127.0.0.1"), "the bound address");
    CHECK(host_allowed("LOCALHOST:8080", "127.0.0.1"), "localhost, any case");
    CHECK(host_allowed("localhost", "127.0.0.1"), "localhost without a port");
    CHECK(host_allowed("[::1]:8080", "127.0.0.1"), "the IPv6 loopback literal");
    CHECK(host_allowed("::1", "127.0.0.1"), "the IPv6 loopback without brackets");
    CHECK(!host_allowed("evil.com", "127.0.0.1"), "a rebinding page's Host");
    CHECK(!host_allowed("evil.com:8080", "127.0.0.1"), "with a port");
    CHECK(!host_allowed("127.0.0.1.evil.com", "127.0.0.1"), "a suffix is not the address");
    CHECK(host_allowed("192.168.1.5:8080", "192.168.1.5"), "the bound LAN address");
    CHECK(!host_allowed("192.168.1.5:8080", "127.0.0.1"), "another machine's address");
    // a wildcard bind cannot name itself: it accepts what reaches it
    CHECK(host_allowed("evil.com", "0.0.0.0"), "a wildcard bind accepts any Host");
    CHECK(host_allowed("localhost", "0.0.0.0"), "localhost too");
    CHECK(host_allowed("[::1]:8080", "::1"), "an IPv6 bind matches its own literal");
}

void test_content_type() {
    CHECK(json_content_type("application/json"), "the media type");
    CHECK(json_content_type("application/json; charset=utf-8"), "with a charset");
    CHECK(json_content_type("Application/JSON;charset=x"), "the type is case-insensitive");
    CHECK(!json_content_type("application/jsonx"), "a prefix is not the media type");
    CHECK(!json_content_type("text/plain"), "a CORS-simple type");
    CHECK(!json_content_type(""), "no header");
    CHECK(!json_content_type("application/xml"), "another type");
}

void test_loopback() {
    CHECK(is_loopback_host("127.0.0.1"), "the default bind");
    CHECK(is_loopback_host("127.0.0.2"), "any 127.0.0.0/8 address");
    CHECK(is_loopback_host("localhost"), "localhost");
    CHECK(is_loopback_host("LOCALHOST"), "localhost, any case");
    CHECK(is_loopback_host("::1") && is_loopback_host("[::1]"), "the IPv6 loopback");
    CHECK(!is_loopback_host("0.0.0.0"), "a wildcard bind");
    CHECK(!is_loopback_host(""), "an empty host is the wildcard");
    CHECK(!is_loopback_host("192.168.1.5"), "a LAN address");
    CHECK(!is_loopback_host("127.evil.com"), "a name that starts like loopback");
    CHECK(!is_loopback_host("128.0.0.1"), "the next block");
    CHECK(!is_loopback_host("127.0.0.256"), "an out-of-range octet");
}

} // namespace

int main() {
    test_bearer();
    test_host();
    test_content_type();
    test_loopback();
    if (omph_test::failures == 0) {
        std::printf("test_server_access: ok\n");
    }
    return omph_test::failures;
}
