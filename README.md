# vani-tensorrt

Host-side TensorRT inference bindings for the
[vāṇी compiler](https://github.com/enthusiasticgeek/vani-compiler), via
`extern "C"` FFI to a hand-written C++ shim. Load a pre-built TensorRT
engine, inspect its input/output tensors by name, and run inference
from vāṇी source.

DLA (NVIDIA's Deep Learning Accelerator, found on Jetson/Orin
hardware) is not a separate binding surface here -- it's a TensorRT
execution-provider config flag set at engine-BUILD time (outside this
package's scope, see "Scope" below), not something this runtime-only
package needs to expose separately.

## Target generation, no backward compatibility

This package targets **TensorRT 10+ (2026-era), full stop** -- there
is no attempt to support older TensorRT releases or compatibility
shims for legacy behavior. Concretely, it binds TensorRT's current
**tensor-name-based execution API**
(`getNbIOTensors`/`getIOTensorName`/`getTensorIOMode`/
`getTensorShape`/`setTensorAddress`/`enqueueV3`) exclusively.
TensorRT's OLDER positional-index "bindings" API
(`getNbBindings`/`bindingIsInput`/`executeV2`) is deprecated as of
TensorRT 8.5 and **removed entirely in TensorRT 10** -- this package
never binds it, in either direction. **If you're on a TensorRT release
older than 10, this package will not link against it as written, and
no porting effort is planned for that direction** -- the tensor-name
API is genuinely more capable (named, order-independent I/O access;
async-by-default execution via an explicit stream) and is where
TensorRT itself is headed, so that's the only surface bound here.

**Older GPUs may or may not work.** Nothing here deliberately excludes
older hardware -- TensorRT 10 itself still supports a wide range of
NVIDIA compute capabilities -- but nothing was written to specifically
verify or support it either. If your GPU or driver predates what your
TensorRT 10 install expects, you'll hit whatever error TensorRT itself
raises through the logger (wired to stderr with a `[vani-tensorrt]`
prefix), not a friendly message from this package. There is no
compatibility layer and none is planned.

## Hardware AND SDK-availability verification status — read this first

This package carries **two** layers of unverified risk, more than
[`vani-cuda`](https://github.com/enthusiasticgeek/vani-cuda) or
[`vani-rocm`](https://github.com/enthusiasticgeek/vani-rocm) had:

1. **No hardware, same as the other two.** No NVIDIA GPU, no TensorRT
   SDK in this development environment.
2. **No SDK available to check against at all, which is new.** Unlike
   the CUDA/HIP Toolkits (both real, apt-installable packages on this
   machine's Debian system, even though not actually installed),
   **TensorRT is not distributed through any standard Linux
   distribution's package manager** -- it requires an NVIDIA Developer
   Program account and a direct download/EULA acceptance. There was no
   way to even attempt installing headers to compile-check this shim
   against real ones.

What *has* been verified directly, without guessing:

- Every `extern "C" fn` declaration in `src/lib.vani` type-checks
  cleanly (`vanic check`).
- `vanic build`/`vanic run --backend=c` generates C that references
  every shim symbol by the exact correct name.
- `shims/vani_tensorrt_shim.cpp` -- which, unlike vani-cuda's/
  vani-rocm's shims, genuinely must be C++ (TensorRT's headers use
  classes and virtual dispatch throughout, no C-callable subset
  exists) -- was confirmed to compile and link correctly as C++
  through vāṇी's *existing* `--link-with <file>.cpp -lstdc++`
  mechanism (gcc/clang auto-select the C++ front end from the `.cpp`
  extension; `-lstdc++` supplies the missing C++ runtime symbols),
  with a synthetic probe file exercising classes/virtual dispatch/
  `new`/`delete`/`std::vector`/`std::string` -- no vāṇी compiler
  changes were needed for this. The shim itself reaches the compiler
  and fails only on the absent `NvInfer.h` header.

If you have a real TensorRT 10+ installation and can spare some time:
please try building against it, and file an issue with whichever
version you tested and what broke, if anything.

## Add to your project

```toml
# vani.toml
[deps]
tensorrt = { registry = "kosh", version = "^0.1" }
```

```sh
vanic add tensorrt
```

## What's included

| Category | Functions |
|---|---|
| Runtime | `trt_create_runtime`, `trt_destroy_runtime` |
| Engine | `trt_load_engine_from_file`, `trt_destroy_engine`, `trt_get_nb_io_tensors`, `trt_get_io_tensor_name`, `trt_tensor_is_input`, `trt_get_tensor_num_elements` |
| Execution context | `trt_create_execution_context`, `trt_destroy_execution_context`, `trt_set_tensor_address` |
| Stream | `trt_create_stream`, `trt_destroy_stream`, `trt_stream_synchronize` |
| Inference | `trt_enqueue` |

**15 functions** -- still fewer than vani-cuda's/vani-rocm's 30+,
because this package's scope is narrower (see "Scope" below) and
because TensorRT's own error-reporting shape (a logger callback, not a
per-call error code) means there's no `trt_error_string`/`trt_check`
equivalent to add -- see the note on `trt_enqueue`'s inverted return
convention below.

Every device pointer, and every runtime/engine/context/stream handle,
crosses the vāṇी boundary as an opaque `i64` -- same convention as
vani-cuda/vani-rocm. Tensors are identified by **name** (`Str`), not
by a positional index -- the modern API's own convention; discover
names via `trt_get_io_tensor_name` before doing anything else with a
given tensor.

**`trt_enqueue`/`trt_set_tensor_address` return the OPPOSITE
convention from every other function here**: 1 = success, 0 = failure
(mirroring TensorRT's own `enqueueV3`/`setTensorAddress`, which return
a `bool`). Every other function returns a handle or count where 0
means "failed to produce one." There's no separate error code on any
failure path -- TensorRT reports problems through a logger callback,
which this package wires to stderr with a `[vani-tensorrt]` prefix.

**Execution is asynchronous by design** -- unlike the older, removed
`executeV2`, `enqueueV3` requires an explicit CUDA stream and returns
immediately. Bind every I/O tensor's address (`trt_set_tensor_address`,
both inputs AND outputs need one), call `trt_enqueue`, then
`trt_stream_synchronize` before reading outputs back to the host.

## Why the shim is C++, not C (unlike vani-cuda/vani-rocm)

TensorRT's public API (`NvInfer.h`) is classes and virtual dispatch
throughout -- `IRuntime`, `ICudaEngine`, `IExecutionContext`,
`ILogger`, all abstract base classes with no C-callable entry points
at all. There is no way to write this shim in plain C the way
vani-cuda's and vani-rocm's Runtime-API shims could. This DOES work
through vāṇी's existing `--link-with` pipeline with no compiler
changes -- see the verification section above for the confirmation --
you just need one extra flag (`-lstdc++`) that a plain-C shim never
needed.

## Building and testing

```sh
# Build your vāṇी program, linking the C++ shim (compiled by vāṇी's
# own $CC, auto-detected as C++ from the .cpp extension) AND the
# TensorRT + CUDA runtime libraries AND the C++ standard library.
vanic build your_program.vani \
  --backend=c \
  --link-with shims/vani_tensorrt_shim.cpp \
  -lnvinfer -lcudart -lstdc++ \
  -o your_program
```

If your TensorRT/CUDA installation isn't on your system `cc`'s default
search path, set the same two environment variables vani-cuda's README
documents (vāṇी's CLI has no `-I`/`-L` flags of its own, but
`cc`/`gcc`/`clang` all honor these):

```sh
export CPATH=/usr/include/x86_64-linux-gnu:/usr/local/cuda/include
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
```

Type-check the bindings alone, without any of the above (no GPU,
TensorRT, or CUDA needed for this step):

```sh
vanic check tests/test_bindings_typecheck.vani
```

## Getting an engine file

This package loads an already-built `.engine`/`.plan` file -- it does
not build one (see "Scope" below). The standard way to produce one
from an ONNX model, using TensorRT's own bundled CLI tool:

```sh
trtexec --onnx=model.onnx --saveEngine=model.engine
```

Engines are tied to the exact TensorRT version and GPU architecture
they were built on/for -- they are not portable the way an ONNX file
is. Rebuild the engine on the target machine if you move to different
hardware or a different TensorRT version.

## Scope

**Inference only, from an already-built engine file.** Deliberately
NOT included in v0.1.0: the Builder + NetworkDefinition +
BuilderConfig + ONNX-parser pipeline that actually *constructs* an
engine from a model. That pipeline is a large, additional, more
version-sensitive C++ API surface, and `trtexec` (which ships with
every TensorRT install) already does this well as a one-time offline
step -- binding it here wouldn't add real capability, only risk and
scope, for a step most users run once per model rather than
per-inference. Composing directly with vani-cuda for device memory
(there's no hard `vani.toml` dependency on it, but you'll need
something that allocates/copies CUDA device memory in practice) is
covered above.

Also out of scope: dynamic input shapes (tensors with a `-1` dimension
aren't supported by `trt_get_tensor_num_elements`, which returns `-1`
for them rather than a usable size -- setting a concrete shape at
execution time via `setInputShape` isn't bound), multi-GPU
orchestration, and INT8 calibration.

## Related packages

[`vani-cuda`](https://github.com/enthusiasticgeek/vani-cuda) is this
package's natural pairing for device memory allocation.
[`vani-rocm`](https://github.com/enthusiasticgeek/vani-rocm) is the
AMD/HIP counterpart to vani-cuda -- TensorRT itself is NVIDIA-only, so
there is no ROCm equivalent of this specific package.

## License

MIT
