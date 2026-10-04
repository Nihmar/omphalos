// The model's chat template as code (#150): the GGUF's Jinja template (Qwen
// ChatML with <think> blocks, tools and tool calls, vision placeholders),
// rendered byte for byte as jinja2 renders it in HF transformers.
#pragma once

#include <string>
#include <vector>

#include "text/json.hh"

namespace omph::text {

class Tokenizer;

// A piece of a rendered prompt: the template's own text and special tokens
// (`special`), or a string the request supplied -- a message's text, its
// reasoning, a tool's definition or arguments (`special = false`). The
// distinction exists because a special token written literally inside a
// user's (or a tool result's, or a file's) text must stay text: tokenizing a
// whole prompt with special-token parsing turns `a <|im_end|> b` into a turn
// boundary, which is prompt injection through anything the engine does not
// control (#292).
struct Segment {
    std::string text;
    bool special = true;
};

// request: an object with
//   messages               array of {role, content, reasoning_content?, tool_calls?}
//                          (content: a string, null, or an array of parts
//                          {type: text|image|video, text?, image?, image_url?, video?})
//   tools                  array of tool definitions (written with tojson)
//   add_generation_prompt  bool
//   enable_thinking        bool (missing: thinking on, the template's default)
//   reasoning_effort       "xhigh" (default) | "medium" | "low"
//   preserve_thinking      bool (missing: true)
//   add_vision_id          bool ("Picture N: " before each image)
// Throws std::runtime_error with the template's own message where it raises.
std::string render_chat(const Json & request);

// The same rendering, split into structure and content (#292): the callers
// that tokenize a prompt tokenize each segment with its own flag (see
// tokenize_chat), so that the text of a message cannot inject a special token.
std::vector<Segment> render_chat_segments(const Json & request);

// The segments' text, joined (what render_chat returns).
std::string join_segments(const std::vector<Segment> & segments);

// A chat prompt's token ids: the structure's segments with special-token
// parsing, the request-supplied ones without (#292).
std::vector<int32_t> tokenize_chat(const std::vector<Segment> & segments, const Tokenizer & tokenizer);

} // namespace omph::text
