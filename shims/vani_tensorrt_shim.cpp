// vani-tensorrt's C++ shim layer: a hand-written wrapper turning
// TensorRT's C++-object-shaped inference API (builder/network/
// engine/execution-context lifecycle, all abstract classes with
// virtual methods) into a flat, opaque-handle, plain-C-ABI surface
// vāṇी's FFI boundary can call -- the SAME goal as vani-cuda's and
// vani-rocm's shims, but a genuinely different SHAPE of problem:
// CUDA/HIP's Runtime APIs are already flat C functions (a thin
// pointer-width-and-error-code translation was enough); TensorRT's
// API has no C entry points at all, so this shim is where the actual
// object lifecycle management happens, not just a type-width cast.
//
// *** MUST be compiled as C++ (unlike vani-cuda/vani-rocm's shims,
// which are deliberately plain C) -- TensorRT's headers use classes,
// namespaces, and virtual dispatch throughout; there is no way to
// avoid this the way dim3-via-aggregate-init let the other two
// packages stay C. Confirmed to work through vāṇी's EXISTING
// `--link-with` pipeline with no compiler changes needed: gcc/clang
// auto-select the C++ front end from the `.cpp` extension, and
// `-lstdc++` (an ordinary `-l<name>` flag vāṇी already supports)
// supplies the missing C++ runtime symbols (operator new/delete,
// RTTI, exception personality routine) at link time. See
// README.md's "Building and testing" section for the exact command.
//
// *** API-VERSION RISK, distinct from (and larger than) the
// "untested on hardware" caveat every hardware-acceleration-tier
// package carries: this shim targets TensorRT's classic bindings-
// based Execution API (`getNbBindings`/`getBindingIndex`/
// `bindingIsInput`/`getBindingDimensions`/`executeV2`), stable from
// roughly TensorRT 7 through 8.x, and the `delete`-based object
// destruction convention TensorRT 8.0+ uses (replacing the older
// `->destroy()` method TensorRT < 8.0 used). TensorRT 10.x
// deprecated/removed several of these bindings-based entry points in
// favor of a newer tensor-NAME-based API (`getNbIOTensors`/
// `getIOTensorName`/`setTensorAddress`/`enqueueV3`). If you're on
// TensorRT 10+, this shim may need porting to that newer API --
// unlike CUDA's/HIP's Runtime APIs (deliberately, famously stable
// across versions), TensorRT's public API has had more churn between
// major versions historically. Written from TensorRT's documented
// API contract; NOT compiled or run against any actual TensorRT
// installation (no TensorRT SDK is available via this development
// environment's package manager at all -- unlike the CUDA/HIP
// toolkits, TensorRT is NVIDIA-account-gated, not distributed through
// standard Linux distro repositories). See README.md's "Hardware AND
// API-version verification status" section before relying on this.
//
// *** SCOPE: inference only. Engine BUILDING (the Builder +
// NetworkDefinition + BuilderConfig + ONNX-parser pipeline, itself a
// large, version-sensitive C++ API surface) is deliberately out of
// scope for v0.1.0 -- the standard, NVIDIA-recommended workflow is to
// build a serialized `.engine`/`.plan` file OFFLINE via `trtexec`
// (TensorRT's own command-line tool, ships with the SDK) from an
// ONNX model, then load that already-built engine at runtime, which
// is exactly what this shim's `trt_load_engine_from_file` does. This
// mirrors vani-algebra's own precedent for a deliberate, documented
// scope-narrowing decision (dropping a hand-derived quartic closed
// form) rather than reaching for a much larger, riskier surface to
// hit a "complete" feeling that isn't actually load-bearing for this
// package's stated purpose.

#include <NvInfer.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace nvinfer1;

namespace {

// A minimal ILogger implementation -- TensorRT requires one to be
// passed to createInferRuntime; there is no default. Only warnings
// and errors are surfaced (to stderr); info/verbose messages are
// dropped, matching a quiet-by-default library convention.
class VaniLogger : public ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[vani-tensorrt] %s\n", msg);
    }
  }
};

VaniLogger &logger() {
  static VaniLogger instance;
  return instance;
}

} // namespace

extern "C" {

// ---- Runtime -----------------------------------------------------------

// Creates the top-level TensorRT runtime. Returns 0 on failure
// (check stderr for the logger's message -- TensorRT has no separate
// error-string query the way CUDA/HIP do; errors are reported through
// the logger callback above).
int64_t trt_create_runtime(void) {
  IRuntime *rt = createInferRuntime(logger());
  return (int64_t)(intptr_t)rt;
}

int32_t trt_destroy_runtime(int64_t runtime) {
  delete (IRuntime *)(intptr_t)runtime;
  return 0;
}

// ---- Engine --------------------------------------------------------------

// Reads a serialized engine file (produced offline, typically via
// `trtexec --onnx=model.onnx --saveEngine=model.engine`) and
// deserializes it. Returns 0 on failure (file not found/unreadable,
// or the blob isn't a valid engine for this TensorRT build/GPU --
// engines are NOT portable across TensorRT versions or GPU
// architectures, check stderr for the logger's message).
int64_t trt_load_engine_from_file(int64_t runtime, const char *path) {
  std::FILE *f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "[vani-tensorrt] could not open engine file: %s\n", path);
    return 0;
  }
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size <= 0) {
    std::fclose(f);
    std::fprintf(stderr, "[vani-tensorrt] engine file is empty: %s\n", path);
    return 0;
  }
  std::vector<char> blob((size_t)size);
  size_t read = std::fread(blob.data(), 1, (size_t)size, f);
  std::fclose(f);
  if (read != (size_t)size) {
    std::fprintf(stderr, "[vani-tensorrt] short read on engine file: %s\n", path);
    return 0;
  }
  IRuntime *rt = (IRuntime *)(intptr_t)runtime;
  ICudaEngine *engine = rt->deserializeCudaEngine(blob.data(), blob.size());
  return (int64_t)(intptr_t)engine;
}

int32_t trt_destroy_engine(int64_t engine) {
  delete (ICudaEngine *)(intptr_t)engine;
  return 0;
}

// Number of bindings (inputs + outputs combined) the engine exposes.
int32_t trt_get_nb_bindings(int64_t engine) {
  return ((ICudaEngine *)(intptr_t)engine)->getNbBindings();
}

// -1 if no binding with this name exists.
int32_t trt_get_binding_index(int64_t engine, const char *name) {
  return ((ICudaEngine *)(intptr_t)engine)->getBindingIndex(name);
}

// 1 if the binding at `index` is an input, 0 if it's an output.
int32_t trt_binding_is_input(int64_t engine, int32_t index) {
  return ((ICudaEngine *)(intptr_t)engine)->bindingIsInput(index) ? 1 : 0;
}

// Total element count for the binding at `index` (product of every
// dimension). Returns -1 if any dimension is dynamic (-1 in
// TensorRT's own Dims representation) -- a dynamic-shape engine needs
// the caller to set the concrete shape at execution time via a
// binding-shape call this v0.1.0 doesn't yet expose (see TODO.md).
int64_t trt_get_binding_num_elements(int64_t engine, int32_t index) {
  Dims dims = ((ICudaEngine *)(intptr_t)engine)->getBindingDimensions(index);
  int64_t total = 1;
  for (int32_t i = 0; i < dims.nbDims; i++) {
    if (dims.d[i] < 0) {
      return -1;
    }
    total *= (int64_t)dims.d[i];
  }
  return total;
}

// ---- Execution context --------------------------------------------------

int64_t trt_create_execution_context(int64_t engine) {
  IExecutionContext *ctx = ((ICudaEngine *)(intptr_t)engine)->createExecutionContext();
  return (int64_t)(intptr_t)ctx;
}

int32_t trt_destroy_execution_context(int64_t context) {
  delete (IExecutionContext *)(intptr_t)context;
  return 0;
}

// Runs synchronous inference. `bindings` is a HOST-side array of
// DEVICE pointers (as i64 handles -- allocate them with vani-cuda's
// cuda_malloc, one per binding index, in binding-index order; upload
// input data with cuda_memcpy_h2d_* before calling this, download
// output data with cuda_memcpy_d2h_* after), `n_bindings` must equal
// trt_get_nb_bindings(engine).
//
// Returns 1 on success, 0 on failure -- NOTE the inverted convention
// relative to every other function in this package (which return an
// i32 handle/count, 0 meaning "failed to produce a handle"): this one
// mirrors TensorRT's own executeV2, which returns a bool, not an
// error/status code. There is no separate error code to inspect on
// failure; check the logger's stderr output.
int32_t trt_execute(int64_t context, const int64_t *bindings, int32_t n_bindings) {
  // A Vec<i64> of device-pointer-sized handles is already laid out
  // exactly like a `void* const*` array on any 64-bit platform (both
  // are contiguous pointer-width values) -- no per-element
  // conversion needed, just a reinterpret of the array itself.
  (void)n_bindings; // only used for documentation/precondition clarity
  bool ok = ((IExecutionContext *)(intptr_t)context)
                ->executeV2((void *const *)(const void *)bindings);
  return ok ? 1 : 0;
}

} // extern "C"
