// llama.cpp's server-side agent tools (#380): what the web UI's tool page
// lists and calls through GET/POST /tools.
//
// The contracts (names, parameter schemas, result shapes, output caps) follow
// llama.cpp's server-tools, so its UI works unchanged. The tools act with the
// server process's permissions: they are off unless --tools/--agent enables
// them, and nothing here sandboxes anything (llama.cpp's --tools-runtime is a
// follow-up).
#pragma once

#include <string>
#include <vector>

#include "text/json.hh"

namespace omph::server::tools {

// The tool names, in llama.cpp's order.
std::vector<std::string> available();

// The enabled list from --tools ("a,b") or "all"; empty means none. Throws
// std::runtime_error naming the available tools when a name is unknown.
std::vector<std::string> parse(const std::string & spec);

// GET /tools: the array the web UI reads (definition, permissions, type).
omph::text::Json list(const std::vector<std::string> & enabled);

// POST /tools: one invocation. `params` is the request's `params` object
// (must be an object, may be null), `cwd` the x-tool-cwd override (empty: the
// server process's directory). Returns the tool's result; a failure of the
// tool itself (a missing file, a timeout) is a `{"error": "..."}` body, not a
// C++ exception. An unknown `name` throws std::runtime_error.
omph::text::Json invoke(const std::string & name, const omph::text::Json & params, const std::string & cwd);

}  // namespace omph::server::tools
