# Vendored chat templates

## Qwen Sharp (`qwen-sharp-v22.5.0.jinja`)

The community template **Qwen Sharp Chat Templates**, vendored as the second
built-in template (#392):

- upstream: https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates
- version: v22.5.0 (`chat_template.jinja`, `template_version =
  "qwen3.8-froggeric-v22.5.0"`)
- built on: froggeric/Qwen-Fixed-Chat-Templates
- license: Apache-2.0 (see `LICENSE-Apache-2.0.txt`); the file is redistributed
  unmodified
- vendored: 2026-10-08

The engine does not run Jinja: `engine/src/text/chat_sharp.cc` is this template
rewritten in C++, kept byte for byte against jinja2 rendering the vendored file
(`tools/check_chat_template.py <model> [<llama-tokenize>] --template sharp`) and
against llama.cpp's tokenizer for the same renderings (#344). `--chat-template
sharp` selects it (`original`, the GGUF's own template, is the default).

To refresh it: download the new `chat_template.jinja` from the pinned version's
tag/commit, replace the vendored file and the templates README (version/date),
update `chat_sharp.cc` for whatever changed, and re-run the check. The
`template_version` in the file is what the byte-for-byte check reports on.
