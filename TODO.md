# vani-tensorrt — TODO

> Scoped in [kosh-index/ROADMAP.md](https://github.com/enthusiasticgeek/kosh-index/blob/main/ROADMAP.md#planned-hardware-acceleration-tier-scoped-2026-08-17-not-started)
> 2026-08-17, implementation started 2026-08-20 (same day as
> [`vani-cuda`](https://github.com/enthusiasticgeek/vani-cuda) and
> [`vani-rocm`](https://github.com/enthusiasticgeek/vani-rocm)). The
> roadmap flagged this as the highest-risk repo of the three. Rewritten
> 2026-08-20 to target TensorRT 10+'s current tensor-name API
> exclusively, no backward compatibility -- READ README.md's "Target
> generation, no backward compatibility" and "Hardware AND
> SDK-availability verification status" before relying on anything
> here; the risk profile is genuinely larger than vani-cuda/vani-rocm's,
> not just the same caveat repeated.

---

## v0.1.0 — Implemented, compile-verified, hardware- AND SDK-UNVERIFIED

### Runtime (2 functions)
- [x] `trt_create_runtime`, `trt_destroy_runtime`

### Engine (6 functions)
- [x] `trt_load_engine_from_file` — loads a pre-built `.engine` file
      (see README's "Getting an engine file")
- [x] `trt_destroy_engine`
- [x] `trt_get_nb_io_tensors`, `trt_get_io_tensor_name`,
      `trt_tensor_is_input`, `trt_get_tensor_num_elements` —
      tensor-NAME-based, replacing the older positional-index
      "bindings" API this package no longer binds at all

### Execution context (3 functions)
- [x] `trt_create_execution_context`, `trt_destroy_execution_context`
- [x] `trt_set_tensor_address` — binds a device pointer to a named
      tensor; call once per I/O tensor before `trt_enqueue`

### Stream (3 functions)
- [x] `trt_create_stream`, `trt_destroy_stream`, `trt_stream_synchronize`
      — `enqueueV3` requires an explicit stream (unlike the older,
      removed `executeV2`, there's no implicit synchronous mode);
      these wrap the CUDA stream calls directly in this shim so the
      package is usable stand-alone without a hard vani-cuda dependency

### Memory (3 functions)
- [x] `trt_memcpy_h2d_f32`, `trt_memcpy_d2h_f32`, `trt_memcpy_d2d_f32`
      — added 2026-08-20, mirroring vani-cuda's/vani-rocm's same-day
      `f32` additions. `f32`-only, deliberately: TensorRT tensors are
      overwhelmingly `f32`, and duplicated directly here (same reason
      as the stream trio above) so the package stays usable
      stand-alone. Device buffer allocation (`cuda_malloc`) is NOT
      duplicated -- it's byte-count-based and dtype-agnostic already,
      nothing TensorRT-specific to add there

### Inference (1 function)
- [x] `trt_enqueue` — asynchronous, via `enqueueV3`; note its inverted
      1=success/0=failure return convention (README explains why)

**Total: 18 vāṇी-facing functions** — still narrower than
vani-cuda/vani-rocm's 36, per the scope decision below (up from an
earlier 11-function bindings-based design, before the 2026-08-20
rewrite to the current tensor-name API added the stream-management
trio and `trt_set_tensor_address`, and the same-day `f32` memcpy
addition brought it to 18).

### Validation performed
- [x] Every declaration in `src/lib.vani` passes `vanic check`
- [x] `tests/test_bindings_typecheck.vani` exercises every single
      binding and type-checks cleanly
- [x] `vanic build --backend=c` (no `--link-with`) confirms codegen
      references every shim symbol by the exact correct name
- [x] `shims/vani_tensorrt_shim.cpp` reaches the system C++ compiler
      (via `--link-with *.cpp -lstdc++`, confirmed to work through
      vāṇी's EXISTING pipeline with no compiler changes -- see the
      synthetic C++-features probe test described in README) and
      fails only on the absent `NvInfer.h` header

### NOT validated (needs real hardware + a real TensorRT 10+ SDK)
- [ ] Any actual TensorRT call whatsoever -- unlike vani-cuda/
      vani-rocm, this shim was never compile-checked against real
      vendor headers at all (TensorRT isn't apt-installable the way
      the CUDA/HIP toolkits were; it requires an NVIDIA Developer
      account)
- [ ] Whether `getNbIOTensors`/`getIOTensorName`/`getTensorIOMode`/
      `getTensorShape`/`setTensorAddress`/`enqueueV3`'s exact
      signatures match a real TensorRT 10+ install byte-for-byte --
      written from documented API contract, never compiled against a
      real header
- [ ] Numerical correctness of any inference run
- [ ] Whether `deserializeCudaEngine`'s 2-argument (no plugin-factory)
      overload assumed here is actually what TensorRT 10+ expects

## Compiler-side finding along the way

Writing this shim surfaced a second real `vanic publish` tarball gap
beyond the one vani-cuda found: the fix that added `.c`/`.h`/`.cu`/
`.cuh` to `copy_dir_vani`'s file filter didn't cover `.cpp`/`.hpp` --
needed here because TensorRT's C++-only API makes a plain-C shim
impossible (unlike CUDA/HIP's C-callable Runtime APIs). Fixed upstream
in `vani-compiler` in the same commit/session, extending the same
filter, with the same regression test extended to cover the two new
extensions.

## Design notes: scope decisions

- **Inference only, pre-built engine only.** The Builder/Network/
  Config/ONNX-parser pipeline that actually constructs an engine is
  deliberately out of scope -- see README's "Scope" section for the
  full reasoning (mirrors vani-algebra's own precedent for a
  documented, deliberate scope-narrowing decision rather than reaching
  for a larger, riskier surface).
- **Tensor-name API only, no backward compatibility.** Rewritten
  2026-08-20 from an initial design that deliberately targeted the
  OLDER bindings-based API (`getNbBindings`/`executeV2`) as a version-
  risk hedge. Per explicit direction to design without backward
  compatibility and assume 2026+ SDKs, that hedge was removed --
  this now binds ONLY `getNbIOTensors`/`getIOTensorName`/
  `getTensorIOMode`/`getTensorShape`/`setTensorAddress`/`enqueueV3`,
  the API generation TensorRT 10+ ships, with no attempt to also
  support the older, now-removed bindings API.
- **No `trt_check`/`trt_error_string` convenience wrapper**, unlike
  vani-cuda's `cuda_check`/vani-rocm's `hip_check`. TensorRT has no
  per-call error CODE to translate -- errors surface through the
  logger callback (wired to stderr) instead. Nothing to wrap.

## Future work (not started, no estimate)

- [ ] Dynamic input shapes (`setInputShape` at execution time)
- [ ] Engine building from ONNX, if `trtexec`-as-an-offline-step turns
      out to be a real friction point for actual users
- [ ] Multiple named profiles / optimization profile selection, if a
      real workload needs it
