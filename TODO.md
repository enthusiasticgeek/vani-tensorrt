# vani-tensorrt — TODO

> Scoped in [kosh-index/ROADMAP.md](https://github.com/enthusiasticgeek/kosh-index/blob/main/ROADMAP.md#planned-hardware-acceleration-tier-scoped-2026-08-17-not-started)
> 2026-08-17, implementation started 2026-08-20 (same day as
> [`vani-cuda`](https://github.com/enthusiasticgeek/vani-cuda) and
> [`vani-rocm`](https://github.com/enthusiasticgeek/vani-rocm)). The
> roadmap flagged this as the highest-risk repo of the three -- READ
> README.md's "Hardware AND API-version verification status" before
> relying on anything here; the risk profile is genuinely larger than
> vani-cuda/vani-rocm's, not just the same caveat repeated.

---

## v0.1.0 — Implemented, compile-verified, hardware- AND SDK-UNVERIFIED

### Runtime (2 functions)
- [x] `trt_create_runtime`, `trt_destroy_runtime`

### Engine (6 functions)
- [x] `trt_load_engine_from_file` — loads a pre-built `.engine` file
      (see README's "Getting an engine file")
- [x] `trt_destroy_engine`
- [x] `trt_get_nb_bindings`, `trt_get_binding_index`,
      `trt_binding_is_input`, `trt_get_binding_num_elements`

### Execution context (2 functions)
- [x] `trt_create_execution_context`, `trt_destroy_execution_context`

### Inference (1 function)
- [x] `trt_execute` — synchronous, via `executeV2`; note its inverted
      1=success/0=failure return convention (README explains why)

**Total: 11 vāṇी-facing functions** — deliberately narrower than
vani-cuda/vani-rocm's 30, per the scope decision below.

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

### NOT validated (needs real hardware + a real TensorRT SDK)
- [ ] Any actual TensorRT call whatsoever -- unlike vani-cuda/
      vani-rocm, this shim was never compile-checked against real
      vendor headers at all (TensorRT isn't apt-installable the way
      the CUDA/HIP toolkits were; it requires an NVIDIA Developer
      account)
- [ ] Whether the targeted API generation (bindings-based
      `executeV2`, `delete`-based object destruction) actually matches
      whatever TensorRT version a real user has -- see README's
      "API-version churn" note. **This is the single biggest open
      question for this package** -- more likely to need a real fix
      than vani-cuda's/vani-rocm's hardware-verification gaps, which
      are "probably fine, needs a hardware run to confirm" rather than
      "may need actual porting work"
- [ ] Numerical correctness of any inference run
- [ ] Whether `deserializeCudaEngine`'s 2-argument (no plugin-factory)
      overload assumed here is actually what the target TensorRT
      version expects (older TensorRT had a 3-argument overload with a
      plugin-factory parameter, removed in newer versions)

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
- **Classic bindings-based API, not the newer tensor-name API.** A
  real, acknowledged bet on which TensorRT generation this targets --
  see README/shim comments. Chosen for being the longer-lived, more
  widely-documented, higher-training-confidence surface, not because
  it's necessarily what a specific user has installed.
- **No `trt_check`/`trt_error_string` convenience wrapper**, unlike
  vani-cuda's `cuda_check`/vani-rocm's `hip_check`. TensorRT has no
  per-call error CODE to translate -- errors surface through the
  logger callback (wired to stderr) instead. Nothing to wrap.

## Future work (not started, no estimate)

- [ ] Port to the tensor-name-based API (`setTensorAddress`/
      `enqueueV3`/`getNbIOTensors`) if TensorRT 10+ compatibility
      turns out to be needed -- likely a real, nontrivial follow-up,
      not a small tweak, given how different the two APIs' calling
      conventions are
- [ ] Dynamic input shapes (`setBindingDimensions` at execution time)
- [ ] Async execution (`enqueueV2`/`enqueueV3` + streams), if a real
      workload needs overlap between transfer and inference
- [ ] Engine building from ONNX, if `trtexec`-as-an-offline-step turns
      out to be a real friction point for actual users
