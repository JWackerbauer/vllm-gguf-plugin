# Vendored llama.cpp sources

Vendored from llama.cpp (<https://github.com/ggml-org/llama.cpp>) at commit a25c986:

- `ggml-common.h`  ← `ggml/src/ggml-common.h` (unmodified)
- `vecdotq.cuh`    ← `ggml/src/ggml-cuda/vecdotq.cuh` (unmodified)

`common.cuh` is a minimal shim that provides the parts of llama.cpp's
`ggml/src/ggml-cuda/common.cuh` that `vecdotq.cuh` depends on. These files are
compiled only in `mmvq_llamacpp.cu`, a separate translation unit, so their
block/table definitions never meet the older copies in `../ggml-common.h`.

llama.cpp is MIT licensed; its license is included as `LICENSE` in this directory.

To update: copy the two files from a newer llama.cpp and rebuild.
