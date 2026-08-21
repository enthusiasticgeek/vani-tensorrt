// vani-tensorrt's C++ shim layer: a hand-written wrapper turning
// TensorRT's C++-object-shaped inference API (builder/network/
// engine/execution-context lifecycle, all abstract classes with
// virtual methods) into a flat, opaque-handle, plain-C-ABI surface
// vāṇী's FFI boundary can call.
//
// *** TARGETS THE CURRENT (2026-era) TENSOR-NAME-BASED EXECUTION
// API ONLY -- getNbIOTensors/getIOTensorName/getTensorIOMode/
// getTensorShape/setTensorAddress/enqueueV3. This is a deliberate
// choice, made explicitly WITHOUT backward compatibility: TensorRT's
// older bindings-based API (getNbBindings/bindingIsInput/executeV2)
// is deprecated as of TensorRT 8.5 and REMOVED in TensorRT 10 --
// there is no attempt here to support both generations or run on an
// installation older than roughly TensorRT 10. If you're on an
// older TensorRT release, this package will not link/work as written
// -- there is no compatibility shim and none is planned. See
// README.md's "Target generation, no backward compatibility" section.
//
// *** MUST be compiled as C++ -- TensorRT's headers use classes,
// namespaces, and virtual dispatch throughout, with no C-callable
// subset. Confirmed to work through vāṇी's EXISTING `--link-with`
// pipeline with no compiler changes needed: gcc/clang auto-select
// the C++ front end from the `.cpp` extension, and `-lstdc++` (an
// ordinary `-l<name>` flag vāṇी already supports) supplies the
// missing C++ runtime symbols. See README.md's "Building and
// testing" section for the exact command.
//
// *** SCOPE: inference only, from an already-built engine file. See
// README.md's "Scope" section for why engine BUILDING (Builder +
// NetworkDefinition + BuilderConfig + ONNX parser) is out of scope --
// `trtexec` (TensorRT's own bundled CLI) already covers that well as
// a one-time offline step.
//
// *** HARDWARE STATUS: written from TensorRT's documented current
// API contract; NOT compiled or run against any actual TensorRT
// installation (no TensorRT SDK is available via this development
// environment's package manager -- unlike CUDA/HIP, TensorRT is
// NVIDIA-account-gated, not distributed through standard Linux distro
// repositories). See README.md's "Hardware verification status".

#include <NvInfer.h>
#include <cuda_runtime.h>
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

// Number of I/O tensors (inputs + outputs combined) the engine
// exposes -- the modern name-based replacement for the old
// getNbBindings().
int32_t trt_get_nb_io_tensors(int64_t engine) {
  return ((ICudaEngine *)(intptr_t)engine)->getNbIOTensors();
}

// The name of the I/O tensor at position `index` (0..nb_io_tensors).
// Every other function below identifies a tensor by this NAME, not a
// position -- discover names by iterating this function first.
const char *trt_get_io_tensor_name(int64_t engine, int32_t index) {
  return ((ICudaEngine *)(intptr_t)engine)->getIOTensorName(index);
}

// 1 if the named tensor is an input, 0 if it's an output.
int32_t trt_tensor_is_input(int64_t engine, const char *name) {
  TensorIOMode mode = ((ICudaEngine *)(intptr_t)engine)->getTensorIOMode(name);
  return (mode == TensorIOMode::kINPUT) ? 1 : 0;
}

// Total element count for the named tensor (product of every
// dimension). Returns -1 if any dimension is dynamic (-1 in
// TensorRT's own Dims representation) -- a dynamic-shape tensor needs
// the caller to set the concrete shape at execution time via a
// binding-shape call this v0.1.0 doesn't yet expose (see TODO.md).
int64_t trt_get_tensor_num_elements(int64_t engine, const char *name) {
  Dims dims = ((ICudaEngine *)(intptr_t)engine)->getTensorShape(name);
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

// Binds a device pointer (as an i64 handle -- allocate it with
// vani-cuda's cuda_malloc or cuda_malloc_async, and for input
// tensors, populate it with input data before calling trt_enqueue)
// to the named tensor. Call this once per I/O tensor (both inputs
// AND outputs need an address set) before every trt_enqueue call
// whose bindings changed. Returns 1 on success, 0 on failure (name
// not found, or the pointer's implied size doesn't match the
// tensor's expected size -- check stderr).
int32_t trt_set_tensor_address(int64_t context, const char *name, int64_t device_ptr) {
  bool ok = ((IExecutionContext *)(intptr_t)context)
                ->setTensorAddress(name, (void *)(intptr_t)device_ptr);
  return ok ? 1 : 0;
}

// ---- Stream --------------------------------------------------------------
//
// enqueueV3 (below) requires an explicit CUDA stream -- unlike the
// old executeV2, there's no implicit "just run synchronously on the
// default stream" mode. These three functions wrap the CUDA stream
// calls directly in this shim (rather than requiring vani-cuda as a
// hard dependency) so this package is usable stand-alone for the
// common case; if you're already using vani-cuda for memory
// management, its own cuda_stream_create/synchronize/destroy work
// identically (both wrap the exact same CUDA Runtime API calls) --
// use whichever is already in scope.

int64_t trt_create_stream(void) {
  cudaStream_t s;
  cudaError_t err = cudaStreamCreate(&s);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[vani-tensorrt] cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return (int64_t)(intptr_t)s;
}

int32_t trt_destroy_stream(int64_t stream) {
  return (int32_t)cudaStreamDestroy((cudaStream_t)(intptr_t)stream);
}

int32_t trt_stream_synchronize(int64_t stream) {
  return (int32_t)cudaStreamSynchronize((cudaStream_t)(intptr_t)stream);
}

// ---- Memory (device-buffer helpers, f32 only) --------------------------
//
// TensorRT tensors are overwhelmingly f32 (float) -- not the i64/f64
// vani-cuda's own memcpy helpers originally shipped with. Added
// 2026-08-20, alongside the matching addition to vani-cuda/vani-rocm,
// specifically so a numerically meaningful vani-tensorrt example can
// be written WITHOUT requiring vani-cuda as a hard dependency (this
// package already duplicates stream management above for the same
// "usable stand-alone" reason). Only f32 is provided here -- i64/f64
// device buffers are a vani-cuda/vani-rocm concern, not a TensorRT-
// tensor concern; if you're already pulling in vani-cuda for other
// reasons, its cuda_malloc/cuda_memcpy_h2d_f32/etc. work identically
// (same underlying cudaMemcpy call) and there's no need for both.
//
// `cuda_malloc`-equivalent allocation for these buffers: use
// vani-cuda's `cuda_malloc`, or note that `cuda_malloc`/`cuda_memset`
// are byte-count based and dtype-agnostic, so any raw device
// allocation works as the destination/source here regardless of
// which package allocated it.

int32_t trt_memcpy_h2d_f32(int64_t dst_device, const float *src_host, int64_t n_elements) {
  return (int32_t)cudaMemcpy((void *)(intptr_t)dst_device, src_host,
                              (size_t)n_elements * sizeof(float),
                              cudaMemcpyHostToDevice);
}

int32_t trt_memcpy_d2h_f32(float *dst_host, int64_t src_device, int64_t n_elements) {
  return (int32_t)cudaMemcpy(dst_host, (const void *)(intptr_t)src_device,
                              (size_t)n_elements * sizeof(float),
                              cudaMemcpyDeviceToHost);
}

int32_t trt_memcpy_d2d_f32(int64_t dst_device, int64_t src_device, int64_t n_elements) {
  return (int32_t)cudaMemcpy((void *)(intptr_t)dst_device, (const void *)(intptr_t)src_device,
                              (size_t)n_elements * sizeof(float),
                              cudaMemcpyDeviceToDevice);
}

// ---- Inference -------------------------------------------------------

// Enqueues inference asynchronously on `stream` (from trt_create_stream,
// or an equivalent vani-cuda cuda_stream_create handle). Every I/O
// tensor must already have its address set via trt_set_tensor_address
// before this call. Returns immediately -- call trt_stream_synchronize
// (or vani-cuda's cuda_stream_synchronize) before reading output
// tensors back to the host.
//
// Returns 1 on SUCCESS, 0 on FAILURE -- note this is the OPPOSITE
// convention from every other function in this package (which return
// a handle/count where 0 means failure): this mirrors TensorRT's own
// enqueueV3, which returns a bool rather than a status code. There is
// no separate error code on failure; check stderr.
int32_t trt_enqueue(int64_t context, int64_t stream) {
  bool ok = ((IExecutionContext *)(intptr_t)context)->enqueueV3((cudaStream_t)(intptr_t)stream);
  return ok ? 1 : 0;
}

} // extern "C"
