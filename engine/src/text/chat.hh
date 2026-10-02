// The model's chat template as code (#150): the GGUF's Jinja template (Qwen
// ChatML with <think> blocks, tools and tool calls, vision placeholders),
// rendered byte for byte as jinja2 renders it in HF transformers.
#pragma once

#include <string>

#include "text/json.hh"

namespace omph::text {

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

} // namespace omph::text
