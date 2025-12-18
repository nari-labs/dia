# MPS (Apple Silicon) Support Implementation Plan

This document identifies all locations in the Dia TTS codebase that need modification to support Apple Silicon MPS (Metal Performance Shaders) backend, and presents an implementation plan for adding MPS support alongside existing CUDA support.

## Executive Summary

The Dia TTS model has **partial MPS support** already implemented. Key pieces like device detection and custom attention implementations exist, but several blocking issues prevent full MPS functionality. This document catalogs all required changes.

**README TODO Item:** "Docker support for ARM architecture and MacOS" (Item #1)

### Important Context: PR #167 Background

[PR #167](https://github.com/nari-labs/dia/pull/167) added `example/simple-mac.py` as a "Mac support" solution, but it was actually a **CPU fallback workaround**, not true MPS support:

> "Avoids MPS backend issues by explicitly using CPU" - PR #167 description

The current `simple-mac.py`:
- Does NOT explicitly set device (defaults to MPS via `_get_default_device()`)
- Uses `float16` (which contradicts `app.py`'s recommendation of `float32` for MPS)
- Only disables `torch.compile` (one of several issues)

**This document aims to achieve true MPS GPU acceleration**, not CPU fallback.

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Blocking Issues](#blocking-issues)
3. [File-by-File Analysis](#file-by-file-analysis)
4. [Implementation Plan](#implementation-plan)
5. [Docker Strategy](#docker-strategy)
6. [Memory & Performance Considerations](#memory--performance-considerations)

---

## Current State Analysis

### What Already Works for MPS

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Device Detection | `dia/model.py:20-25` | ✅ Working | Correctly detects CUDA → MPS → CPU |
| Device Detection | `app.py:27-36` | ✅ Working | Includes MPS check |
| Custom Attention | `dia/layers.py:139-189` | ✅ Working | `custom_scaled_dot_product_attention()` for MPS |
| CrossAttention MPS | `dia/layers.py:284-305` | ✅ Working | Routes to custom attention on MPS |
| SelfAttention MPS | `dia/layers.py:502-523` | ✅ Working | Routes to custom attention on MPS |
| RMSNorm dtype | `dia/layers.py` (multiple) | ✅ Working | Always uses float32 (stable for MPS) |
| Gradio dtype | `app.py:43-50` | ✅ Working | Maps MPS → float32 |
| Gradio compile | `app.py:178` | ✅ Working | `use_torch_compile=False` |
| CUDA TF32 guard | `dia/model.py:128-129` | ✅ Working | Only runs on CUDA |
| CUDA seed guard | `cli.py:17-22`, `app.py:62-66` | ✅ Working | CUDA-specific code guarded |

### What Needs Fixing

| Issue | File | Line(s) | Severity | Description |
|-------|------|---------|----------|-------------|
| CUDAGraph call | `dia/model.py` | 701 | 🔴 Critical | Called unconditionally, will error/warn on MPS |
| torch.compile MPS | `dia/model.py` | 656-660 | 🔴 Critical | No device check before compiling |
| CLI device default | `cli.py` | 72-74 | 🟡 Medium | Missing MPS in default device selection |
| Example dtype | `example/simple-mac.py` | 4 | 🟡 Medium | Uses float16 instead of float32 |
| Triton config | `example/benchmark.py` | 8-10 | 🟡 Medium | Unguarded Triton-specific config |
| MPS seed | `cli.py`, `app.py` | - | 🟢 Low | No MPS-specific seed management |
| Docker ARM | `docker/` | - | 🟡 Medium | No ARM/macOS Docker support |

---

## Blocking Issues - Deep Analysis

### Issue 1: Unconditional CUDAGraph Call (Critical)

**File:** `dia/model.py`
**Line:** 701
**Severity:** 🔴 Critical

#### What is CUDA Graph?

[CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) are a NVIDIA technology that captures a sequence of GPU operations (kernel launches) and replays them as a single unit. This provides significant performance benefits:

1. **Reduced CPU-GPU Context Switching** - Each kernel launch normally requires CPU involvement
2. **Lower Kernel Launch Latency** - Eliminates per-operation driver overhead
3. **Optimized Memory Management** - Tensors from prior iterations are freed efficiently

#### Why is it Used in Dia's Generation Loop?

Looking at the code context in `model.py:695-760`:

```python
# --- Generation Loop ---
while dec_step < max_tokens:
    # ...
    current_step_idx = dec_step + 1
    torch.compiler.cudagraph_mark_step_begin()  # <-- Line 701
    dec_state.prepare_step(dec_step)
    tokens_Bx1xC = dec_output.get_tokens_at(dec_step).repeat_interleave(2, dim=0)

    pred_BxC = self._decoder_step(...)  # The main decoder computation
```

**The Role in Dia:**
- Dia generates audio tokens **autoregressively** - one token at a time
- For a 10-second audio clip, the loop runs **~860 iterations** (1 second ≈ 86 tokens)
- Each iteration calls `self._decoder_step()` which runs the full decoder model
- Without CUDA Graphs, each iteration incurs kernel launch overhead
- `cudagraph_mark_step_begin()` signals PyTorch that a new iteration is starting, allowing efficient memory reuse

**Performance Impact (from README benchmarks):**
| Mode | Realtime Factor |
|------|-----------------|
| CUDA + torch.compile (uses CUDA Graphs) | ~2.2x |
| CUDA without torch.compile | ~1.3x |
| **Speedup from CUDA Graphs** | **~70%** |

#### What Happens on MPS?

MPS (Metal Performance Shaders) has **no equivalent to CUDA Graphs**. When this function is called on MPS:

| Scenario | Behavior |
|----------|----------|
| Best case | No-op (function does nothing) |
| Likely case | Warning printed each iteration (~860 times) |
| Worst case | RuntimeError crashes generation |

Testing is needed to confirm actual behavior, but the function is explicitly CUDA-specific per [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.compiler.cudagraph_mark_step_begin.html).

#### Relationship to torch.compile

This function works **in conjunction with** `torch.compile(mode="max-autotune")` on line 659:
- `torch.compile` captures the operations into a graph
- `cudagraph_mark_step_begin()` marks iteration boundaries for memory management
- Both must be disabled/guarded for MPS support

#### Researched Solution

**Online Research Findings:**

1. **No Built-in Guard:** The [cudagraph_trees.py source](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/cudagraph_trees.py) shows that `cudagraph_mark_step_begin()` assumes CUDA availability - it has no runtime device checks.

2. **Standard Pattern:** The [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html) and community best practices recommend using `device.type == "cuda"` checks before CUDA-specific operations.

3. **Alternative Approaches Considered:**
   - `torch.cuda.is_available()` - Less precise, checks global availability not current device
   - `torch.backends.cuda.is_built()` - Checks build config, not runtime device
   - **Best: `self.device.type == "cuda"`** - Checks the actual device in use

**Recommended Fix:**

```python
# Current code (problematic - line 701)
torch.compiler.cudagraph_mark_step_begin()

# Fixed code
if self.device.type == "cuda":
    torch.compiler.cudagraph_mark_step_begin()
```

**Why This Works:**
- `self.device` is already stored in the `Dia` class (set at `model.py:116`)
- The check is minimal overhead (simple string comparison)
- Skipping the call on non-CUDA devices is safe - it's purely an optimization hint
- Generation proceeds normally, just without CUDA Graph memory optimization

**References:**
- [torch.compiler.cudagraph_mark_step_begin() docs](https://docs.pytorch.org/docs/stable/generated/torch.compiler.cudagraph_mark_step_begin.html)
- [PyTorch MPS Backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal PyTorch Guide](https://developer.apple.com/metal/pytorch/)

---

### Issue 2: torch.compile Without Device Check (Critical)

**File:** `dia/model.py`
**Lines:** 656-660
**Severity:** 🔴 Critical

#### What is torch.compile?

[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) is PyTorch 2.0's flagship feature - a JIT (Just-In-Time) compiler that:

1. **Captures** Python code as computational graphs (via TorchDynamo)
2. **Optimizes** the graphs (via TorchInductor)
3. **Generates** optimized code for the target hardware

#### The Compilation Modes

| Mode | Description | Backend |
|------|-------------|---------|
| `default` | Basic optimization | Platform-specific |
| `reduce-overhead` | Uses CUDA Graphs | CUDA only |
| `max-autotune` | Autotuning + CUDA Graphs + Triton kernels | CUDA/ROCm only |

#### How Dia Uses torch.compile

Looking at `model.py:656-660`:

```python
if use_torch_compile and not hasattr(self, "_compiled"):
    # Compilation can take about a minute.
    self._prepare_generation = torch.compile(self._prepare_generation, dynamic=True, fullgraph=True)
    self._decoder_step = torch.compile(self._decoder_step, fullgraph=True, mode="max-autotune")
    self._compiled = True
```

**Two functions are compiled:**

| Function | Mode | Why |
|----------|------|-----|
| `_prepare_generation` | `dynamic=True` | Input sizes vary (different text lengths) |
| `_decoder_step` | `max-autotune` | Fixed shapes, hot path (called ~860 times) |

#### Why "max-autotune" is Critical for Performance

The `max-autotune` mode:

1. **Uses [Triton](https://openai.com/index/triton/)** to generate optimized GPU kernels
2. **Enables CUDA Graphs** by default (hence the need for `cudagraph_mark_step_begin()`)
3. **Autotuning** - tests multiple kernel implementations and selects the fastest

Per the README benchmarks:
- With `max-autotune`: **~2.2x realtime**
- Without: **~1.3x realtime**

#### Why This Fails on MPS

**Triton is CUDA/ROCm exclusive:**

- Triton generates PTX (NVIDIA) or GCN (AMD) assembly code
- MPS uses [Metal Shaders](https://developer.apple.com/metal/pytorch/) - Apple's completely different GPU programming model
- There is **no Triton backend for MPS** and no plans to add one

**The error flow:**

```
User calls generate(use_torch_compile=True) on MPS
  ↓
torch.compile(_prepare_generation, ...) ← May produce warnings
  ↓
torch.compile(_decoder_step, mode="max-autotune") ← FAILS (Triton missing)
  ↓
RuntimeError or backend error
```

#### Note About CPU

CPU also doesn't benefit from `max-autotune`. The fix should check for CUDA specifically.

#### Researched Solution

**Online Research Findings:**

1. **MPS torch.compile Status ([Issue #150121](https://github.com/pytorch/pytorch/issues/150121)):**
   - Currently in "early prototype stage" (as of March 2025)
   - Target: "tentative beta status for PyTorch 2.8.0"
   - "Attempt to use it to accelerate end-to-end network is likely to fail"
   - Known issues: FlexAttention not implemented, dynamic shapes missing, reduction performance worse than eager

2. **The Error on MPS:**
   ```
   AssertionError: Device mps not supported
   ```
   When using `mode="max-autotune"` with Inductor backend.

3. **Alternative Modes Considered:**
   - `"default"` mode - May work on MPS but still experimental
   - `"max-autotune-no-cudagraphs"` - Skips CUDA Graphs but still needs Triton (fails on macOS)
   - **Conclusion:** No compilation mode works reliably on MPS currently

4. **Workaround Options:**
   - `torch._dynamo.config.suppress_errors = True` - Not recommended, hides issues
   - `PYTORCH_ENABLE_MPS_FALLBACK=1` - Falls back to CPU for unsupported ops (slow)
   - **Best: Skip compilation on non-CUDA** - Clean, predictable behavior

**Recommended Fix:**

```python
# Current code (problematic)
if use_torch_compile and not hasattr(self, "_compiled"):
    self._prepare_generation = torch.compile(self._prepare_generation, dynamic=True, fullgraph=True)
    self._decoder_step = torch.compile(self._decoder_step, fullgraph=True, mode="max-autotune")
    self._compiled = True

# Fixed code
if use_torch_compile and not hasattr(self, "_compiled"):
    if self.device.type != "cuda":
        import warnings
        warnings.warn(
            f"torch.compile with max-autotune is only supported on CUDA devices. "
            f"Current device: {self.device.type}. Skipping compilation."
        )
    else:
        self._prepare_generation = torch.compile(self._prepare_generation, dynamic=True, fullgraph=True)
        self._decoder_step = torch.compile(self._decoder_step, fullgraph=True, mode="max-autotune")
        self._compiled = True
```

**Why This Works:**
- Single check at compilation time (not per-iteration)
- Warning is informative but not disruptive
- Generation still works, just without JIT optimization
- Future PyTorch versions with MPS torch.compile support will work automatically when `device.type` check is updated

**References:**
- [torch.compile MPS Progress Tracker (#150121)](https://github.com/pytorch/pytorch/issues/150121)
- [torch.compile Documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [MPS Backend Documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)

---

### Issue 3: CLI Device Default Missing MPS

**File:** `cli.py`
**Lines:** 70-75
**Severity:** 🟡 Medium

#### The Problem

The CLI tool's device argument uses a binary check:

```python
infra_group.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",  # <-- No MPS!
    help="Device to run inference on (e.g., 'cuda', 'cpu', default: auto).",
)
```

#### Inconsistency with Other Entry Points

**`app.py:26-36`** (Gradio UI) - ✅ Correct:

```python
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")  # ✅ MPS included!
else:
    device = torch.device("cpu")
```

**`model.py:20-25`** (`_get_default_device()`) - ✅ Correct:

```python
def _get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # ✅ MPS included!
    return torch.device("cpu")
```

#### User Experience Impact

A Mac user running the CLI:

```bash
python cli.py "[S1] Hello world." --output test.wav
```

| Expected Behavior | Actual Behavior |
|-------------------|-----------------|
| Uses MPS (GPU acceleration) | Falls back to CPU |
| Fast generation | Slow generation |
| Warning if MPS unavailable | Silent degradation |

The user has to **manually specify** `--device mps` to get GPU acceleration, which is non-obvious.

#### Why This Matters for MPS Support

The CLI is a primary entry point for:

1. **Testing** - Developers verifying MPS works
2. **Scripting** - Automated audio generation pipelines
3. **Quick usage** - One-off audio generation

If the default is wrong, users may conclude "MPS doesn't work" when it actually would work if specified.

#### Researched Solution

**Online Research Findings:**

1. **[Official PyTorch CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)** recommends using argparse with `is_available()` checks.

2. **[Modern Best Practice Pattern](https://mctm.web.id/blog/2024/PyTorchGPUSelect/)** uses a helper function that checks CUDA → MPS → CPU in priority order.

3. **[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/common/trainer.html)** supports `accelerator="auto"` which auto-detects the best device.

4. **Key Pattern Elements:**
   - Check `torch.cuda.is_available()` first (NVIDIA GPU)
   - Check `torch.backends.mps.is_available()` second (Apple Silicon)
   - Fall back to CPU
   - Allow manual override via argparse

**Recommended Fix:**

```python
# Add helper function at module level (after imports)
def _get_default_device_str():
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Update argparse definition
infra_group.add_argument(
    "--device",
    type=str,
    default=_get_default_device_str(),
    help="Device to run inference on ('cuda', 'mps', 'cpu', or 'auto', default: auto-detect).",
)
```

**Alternative: Add 'auto' as explicit option:**

```python
infra_group.add_argument(
    "--device",
    type=str,
    default="auto",
    choices=["auto", "cuda", "mps", "cpu"],
    help="Device to run inference on (default: auto-detect).",
)

# Then in main code:
if args.device == "auto":
    device = torch.device(_get_default_device_str())
else:
    device = torch.device(args.device)
```

**Why This Pattern:**

- Consistent with `app.py` and `model.py` behavior
- User gets GPU acceleration without knowing device names
- Manual override still possible for debugging
- `hasattr` check ensures compatibility with older PyTorch versions

**References:**
- [PyTorch CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [Simplifying PyTorch Device Selection](https://mctm.web.id/blog/2024/PyTorchGPUSelect/)
- [PyTorch MPS Availability Check](https://discuss.pytorch.org/t/how-to-check-mps-availability/152015)

---

### Issue 4: simple-mac.py Uses float16 Instead of float32

**File:** `example/simple-mac.py`
**Line:** 4
**Severity:** 🟡 Medium

#### The Problem

The Mac example script uses float16:

```python
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
```

But the Gradio UI (`app.py:43-49`) explicitly recommends float32 for MPS:

```python
dtype_map = {
    "cpu": "float32",
    "mps": "float32",  # Apple M series – better with float32
    "cuda": "float16",  # NVIDIA – better with float16
}
```

#### Understanding MPS dtype Support

| dtype | MPS Support | Performance | Notes |
|-------|-------------|-------------|-------|
| float32 | ✅ Full | Baseline | Recommended for stability |
| float16 | ✅ Supported | ~0-30% faster | May have precision issues |
| bfloat16 | ❌ **Not supported** | N/A | Will error on MPS |
| float64 | ❌ **Not supported** | N/A | Will error on MPS |

#### Why float16 Provides Minimal Benefit on MPS

**NVIDIA GPUs:**
- Have dedicated **Tensor Cores** optimized for float16 operations
- Can achieve **2-4x speedup** with float16 vs float32

**Apple Silicon:**
- Lacks dedicated float16 acceleration hardware
- [Research shows](https://arxiv.org/pdf/2501.14925) only **~0-30% performance gain** with float16
- The ANE (Apple Neural Engine) doesn't integrate with PyTorch MPS

#### Historical float16 Issues on MPS

There have been [documented bugs](https://github.com/pytorch/pytorch/issues/78168) with float16 precision on MPS:

- Values converting incorrectly (e.g., 0.4495 → 0.0847)
- Numerical instability in certain operations
- Most bugs are fixed in recent PyTorch versions, but occasional issues remain

#### Why This Matters for Audio Generation

Audio generation is **sensitive to numerical precision**:

1. **DAC (Descript Audio Codec)** encodes/decodes audio waveforms
2. **Small numerical errors** can cause audible artifacts
3. **Accumulated errors** over ~860 autoregressive steps can degrade quality

Using float32 trades minor performance for **guaranteed audio quality**.

#### Memory Comparison

| dtype | Model Size (1.6B params) | With 96GB RAM |
|-------|-------------------------|---------------|
| float16 | ~3.2 GB | ✅ No issue |
| float32 | ~6.4 GB | ✅ No issue |

With 96GB unified memory, there's no reason to sacrifice precision for memory savings.

#### Researched Solution

**Online Research Findings:**

1. **[HuggingFace Diffusers MPS Guide](https://huggingface.co/docs/diffusers/optimization/mps):**
   - Recommends `torch_dtype=torch.float16` for Diffusers BUT with attention slicing
   - Notes MPS is "very sensitive to memory pressure"
   - Uses float16 primarily for memory savings, not speed

2. **[Apple Metal PyTorch Guide](https://developer.apple.com/metal/pytorch/):**
   - Default recommendation: Use float32
   - MPS supports float16 but "prefer float32 for compatibility"

3. **[PyTorch MPS Backend Notes](https://docs.pytorch.org/docs/stable/notes/mps.html):**
   - float16 supported but historically buggy
   - bfloat16 NOT supported (common error source)
   - float64 NOT supported

4. **[Community Testing (Mac ML Speed Test)](https://github.com/mrdbourke/mac-ml-speed-test):**
   - Float16 doesn't halve training times on MPS like it does on NVIDIA
   - Float32 is used for all MPS benchmarks for consistency

5. **Industry Practice (Flux/ComfyUI on Apple Silicon):**
   - Explicitly uses `--force-fp16` flag for memory efficiency
   - But acknowledges quality vs speed tradeoff

**Recommended Fix:**

```python
# Current code (simple-mac.py:4) - problematic
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")

# Fixed code
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float32")
```

**Additional Recommendations:**

- Also update the comment in simple-mac.py to explain the dtype choice
- Consider making the example explicitly set device too: `device=torch.device("mps")`

**References:**
- [HuggingFace Diffusers MPS Guide](https://huggingface.co/docs/diffusers/optimization/mps)
- [Apple Metal PyTorch Guide](https://developer.apple.com/metal/pytorch/)
- [PyTorch MPS Backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch MPS Float16 Issue #78168](https://github.com/pytorch/pytorch/issues/78168)

---

### Issue 5: Triton Config in benchmark.py

**File:** `example/benchmark.py`
**Lines:** 8-10
**Severity:** 🟡 Medium

#### The Problem

The benchmark script sets Triton-specific configuration unconditionally:

```python
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
```

#### What These Settings Do

| Setting | Purpose | Triton Dependency |
|---------|---------|-------------------|
| `coordinate_descent_tuning` | Optimize kernel parameters | Yes |
| `triton.unique_kernel_names` | Unique names for debugging | Yes (explicit) |
| `fx_graph_cache` | Cache compiled graphs | Partial |

#### Why This Fails on macOS

Triton is only available on Linux and Windows (per `pyproject.toml:21-22`):

```toml
"triton==3.2.0 ; sys_platform == 'linux'",
"triton-windows==3.2.0.post18 ; sys_platform == 'win32'",
```

On macOS:

- Triton is **not installed**
- Accessing `torch._inductor.config.triton` may raise `AttributeError`
- Or produce warnings about missing Triton backend

#### Researched Solution

**Online Research Findings:**

1. **[PyTorch Inductor Config Source](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py):**
   - Uses `sys.platform == "darwin"` for macOS-specific C++ compiler selection
   - Triton config class has NO built-in platform guards
   - Relies on Triton's own internal platform detection

2. **[Triton Installation Issue #125093](https://github.com/pytorch/pytorch/issues/125093):**
   - Error: `RuntimeError: Cannot find a working triton installation`
   - Workaround: `torch._dynamo.config.suppress_errors = True` (not recommended)

3. **[vLLM macOS Issue #28352](https://github.com/vllm-project/vllm/issues/28352):**
   - "Triton not installed or not compatible" on Apple Silicon
   - Confirms Triton is Linux/Windows only

4. **Platform Check Pattern in PyTorch:**
   ```python
   # From torch/_inductor/config.py
   "clang++" if sys.platform == "darwin" else "g++"
   ```
   This is the standard pattern for platform-conditional code.

**Recommended Fix:**

```python
import sys

# Only configure Triton on platforms where it's available
if sys.platform in ("linux", "win32"):
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True
```

**Alternative: Use try/except for robustness:**

```python
try:
    # Only works when Triton is available (Linux/Windows)
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True
except (AttributeError, RuntimeError):
    # Triton not available on this platform (macOS)
    pass
```

**References:**
- [PyTorch Inductor Config Source](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)
- [Triton Installation Error (#125093)](https://github.com/pytorch/pytorch/issues/125093)
- [Simplify Triton Error (#143406)](https://github.com/pytorch/pytorch/issues/143406)

---

### Issue 6: MPS Seed Management (Optional)

**Files:** `cli.py`, `app.py`
**Severity:** 🟢 Low

#### Current Seed Management

Both files have CUDA-specific seed handling:

```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

#### MPS Seed Support

PyTorch provides `torch.mps.manual_seed()` for MPS devices:

```python
if torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)
```

#### Why This is Low Priority

1. `torch.manual_seed()` sets the global seed, which **already affects MPS**
2. `torch.mps.manual_seed()` is primarily for explicit control
3. The current code **works** - it just doesn't have explicit MPS handling

#### Researched Solution

**Online Research Findings:**

1. **[PyTorch torch.mps Documentation](https://docs.pytorch.org/docs/stable/mps.html):**
   - `torch.mps.manual_seed(seed)` - Sets seed for MPS random number generation
   - `torch.mps.seed()` - Sets to random number
   - `torch.mps.get_rng_state()` / `torch.mps.set_rng_state()` - State management
   - **Note:** No `manual_seed_all()` equivalent exists for MPS

2. **[PyTorch Reproducibility Guide](https://docs.pytorch.org/docs/stable/notes/randomness.html):**
   - `torch.manual_seed()` affects all devices including MPS
   - Results may not be reproducible between CPU and GPU (even with same seed)
   - Some operations are inherently non-deterministic

3. **[MPS Dropout Non-Determinism (Issue #84516)](https://github.com/pytorch/pytorch/issues/84516):**
   - Dropout is non-deterministic on MPS even with manual seed set
   - Workaround: Sample on CPU and move to MPS

4. **[PyTorch Lightning MPS Seed Issue (#20145)](https://github.com/Lightning-AI/pytorch-lightning/issues/20145):**
   - `local_torch_manual_seed` doesn't work with MPS tensors
   - Community-reported reproducibility challenges

**Recommended Enhancement:**

```python
def set_seed(seed: int):
    """Sets random seed for reproducibility across all backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA-specific seeding
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # MPS-specific seeding (Apple Silicon)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # cuDNN determinism (only affects CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Important Caveats:**

- MPS reproducibility is NOT guaranteed even with proper seeding
- Some operations (like Dropout) may still be non-deterministic
- For strict reproducibility, consider CPU execution

**References:**
- [torch.mps Documentation](https://docs.pytorch.org/docs/stable/mps.html)
- [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [MPS Dropout Issue (#84516)](https://github.com/pytorch/pytorch/issues/84516)

---

### Issue 7: Docker ARM Support

**Location:** `docker/`
**Severity:** 🟡 Medium

#### Current Docker State

| File | Architecture | GPU Support |
|------|--------------|-------------|
| `Dockerfile.gpu` | x86_64 | NVIDIA CUDA |
| `Dockerfile.cpu` | x86_64 | None |

#### The README TODO Item

> "Docker support for ARM architecture and MacOS" (Item #1)

#### Critical Understanding: Docker Cannot Access MPS

**Docker on macOS cannot use MPS GPU acceleration:**

1. Docker containers run in a Linux VM (even on ARM Macs)
2. MPS requires direct access to Apple's Metal framework
3. Apple does not provide GPU passthrough for containers

This means:
- **ARM Linux Docker** = CPU-only (for AWS Graviton, Raspberry Pi, etc.)
- **macOS MPS** = Native installation only (no Docker)

#### Recommended Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Support Matrix                     │
├─────────────────┬──────────────┬───────────────────────────┤
│ Platform        │ Architecture │ Docker Support            │
├─────────────────┼──────────────┼───────────────────────────┤
│ Linux + NVIDIA  │ x86_64       │ ✅ Dockerfile.gpu (CUDA)  │
│ Linux CPU       │ x86_64       │ ✅ Dockerfile.cpu         │
│ Linux CPU       │ ARM64        │ 🆕 Dockerfile.arm (new)   │
│ macOS + MPS     │ ARM64        │ ❌ Native only (no Docker)│
└─────────────────┴──────────────┴───────────────────────────┘
```

#### Proposed Dockerfile.arm

```dockerfile
# Dockerfile.arm - ARM64 Linux deployment for DIA (CPU-only)
# Build: docker buildx build --platform linux/arm64 -f docker/Dockerfile.arm -t dia-arm .
# Run:   docker run --rm -p 7860:7860 dia-arm

FROM --platform=linux/arm64 python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-venv libsndfile1 ffmpeg curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 appuser && \
    mkdir -p /app && chown -R appuser:appuser /app

USER appuser
WORKDIR /app
COPY --chown=appuser:appuser . .

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# ARM64 PyTorch (CPU-only, no Triton)
RUN pip install --upgrade pip && \
    pip install torch torchaudio && \
    pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app GRADIO_SERVER_NAME="0.0.0.0"
EXPOSE 7860
CMD ["python3", "app.py"]
```

#### Documentation for Native macOS MPS

For macOS MPS users, provide clear installation instructions:

```bash
# Native macOS installation for MPS support
git clone https://github.com/nari-labs/dia.git
cd dia

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (will use default PyPI, which includes MPS support)
pip install -e .

# Run with MPS
python app.py  # Automatically detects MPS
```

#### Researched Solution

**Online Research Findings:**

1. **Docker buildx Multi-Architecture Builds:**
   - Docker buildx enables building images for multiple architectures (linux/amd64, linux/arm64) from a single command
   - Use `docker buildx create --use` to create a builder instance
   - Build command: `docker buildx build --platform linux/amd64,linux/arm64 -t image:tag .`
   - Can push multi-arch manifests to registries with `--push` flag

2. **[PyTorch ARM64 Docker Image Issue (#81224)](https://github.com/pytorch/pytorch/issues/81224):**
   - Community request for official MPS-ready ARM64 Docker images
   - **Status:** Closed as not planned (June 2023)
   - **Key insight:** Docker runs Linux VMs on macOS, so MPS access is fundamentally impossible
   - Official recommendation: Native installation for MPS acceleration

3. **ARM64 Linux PyTorch Availability:**
   - PyTorch provides ARM64 Linux wheels on PyPI
   - Works on: AWS Graviton, Ampere Altra, Raspberry Pi 4/5, Apple Silicon (Linux VMs)
   - Install: `pip install torch torchaudio` (auto-detects ARM64)
   - No Triton support on ARM64 Linux

4. **Multi-Architecture Docker Best Practices:**
   - Use `--platform` flag in FROM statements for clarity
   - Separate Dockerfiles for significantly different builds (GPU vs CPU vs ARM)
   - Use GitHub Actions with `docker/setup-qemu-action` for CI/CD multi-arch builds
   - Consider manifest lists for registry distribution

**Recommended Multi-Architecture Build Commands:**

```bash
# Create buildx builder (one-time setup)
docker buildx create --name dia-builder --use

# Build ARM64 image
docker buildx build --platform linux/arm64 \
    -f docker/Dockerfile.arm \
    -t dia:arm64 \
    --load .

# Build multi-arch and push to registry
docker buildx build --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile.cpu \
    -t myregistry/dia:latest \
    --push .
```

**GitHub Actions CI/CD Example:**

```yaml
name: Build Multi-Arch Docker Images

on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-qemu-action@v3

      - uses: docker/setup-buildx-action@v3

      - uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.cpu
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
```

**Important Clarifications:**

- **macOS + MPS:** Cannot use Docker. Must install natively.
- **ARM64 Linux (Graviton, etc.):** CPU-only via Dockerfile.arm
- **x86_64 Linux + NVIDIA:** Full CUDA support via Dockerfile.gpu
- **x86_64 Linux CPU:** Via Dockerfile.cpu

**References:**
- [Docker buildx documentation](https://docs.docker.com/buildx/working-with-buildx/)
- [PyTorch ARM64 Docker Issue (#81224)](https://github.com/pytorch/pytorch/issues/81224)
- [GitHub Actions - docker/setup-buildx-action](https://github.com/docker/setup-buildx-action)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

---

## File-by-File Analysis

### `dia/model.py` - Core Model

| Line(s) | Code | Status | Action Required |
|---------|------|--------|-----------------|
| 20-25 | `_get_default_device()` | ✅ Good | None |
| 79-92 | `ComputeDtype` enum | ✅ Good | None |
| 116 | Device assignment | ✅ Good | None |
| 128-129 | CUDA TF32 setting | ✅ Good | Already guarded |
| 163, 170, 215 | Device placement | ✅ Good | Properly handled |
| 656-660 | `torch.compile` | ❌ Fix | Add MPS device check |
| 701 | `cudagraph_mark_step_begin` | ❌ Fix | Add CUDA-only guard |

### `dia/layers.py` - Neural Network Layers

| Line(s) | Code | Status | Action Required |
|---------|------|--------|-----------------|
| 139-189 | `custom_scaled_dot_product_attention()` | ✅ Good | MPS-compatible implementation |
| 284-305 | CrossAttention MPS branch | ✅ Good | Correctly routes to custom fn |
| 502-523 | SelfAttention MPS branch | ✅ Good | Correctly routes to custom fn |
| 541-560 | RMSNorm (Encoder) | ✅ Good | Uses float32 |
| 606-610 | RMSNorm (Encoder final) | ✅ Good | Uses float32 |
| 639-652 | RMSNorm (Decoder) | ✅ Good | Uses float32 |
| 750-754 | RMSNorm (Decoder final) | ✅ Good | Uses float32 |

### `cli.py` - Command Line Interface

| Line(s) | Code | Status | Action Required |
|---------|------|--------|-----------------|
| 12-22 | `set_seed()` | ✅ Good | CUDA parts guarded |
| 72-74 | Device default | ❌ Fix | Add MPS to default |
| 96 | Device assignment | ✅ Good | Uses argparse value |

### `app.py` - Gradio Web UI

| Line(s) | Code | Status | Action Required |
|---------|------|--------|-----------------|
| 27-36 | Device detection | ✅ Good | Includes MPS |
| 43-50 | `dtype_map` | ✅ Good | MPS → float32 |
| 57-66 | `set_seed()` | ✅ Good | CUDA parts guarded |
| 178 | `use_torch_compile` | ✅ Good | Disabled |

### `example/simple-mac.py` - Mac Example

| Line(s) | Code | Status | Action Required |
|---------|------|--------|-----------------|
| 4 | `compute_dtype="float16"` | ❌ Fix | Change to "float32" |
| 10 | `use_torch_compile=False` | ✅ Good | Correctly disabled |

### `example/benchmark.py` - Benchmarking

| Line(s) | Code | Status | Action Required |
|---------|------|--------|-----------------|
| 8-10 | Triton config | ❌ Fix | Guard with platform check |
| 16 | `compute_dtype` | ⚠️ Note | CUDA-optimized (expected) |

### `pyproject.toml` - Dependencies

| Line(s) | Code | Status | Action Required |
|---------|------|--------|-----------------|
| 21-22 | Triton (Linux/Windows only) | ✅ Good | Correctly excluded from macOS |
| 49-55 | UV CUDA sources | ⚠️ Note | macOS uses default PyPI |

### `docker/Dockerfile.gpu` - GPU Docker

| Line(s) | Code | Status | Action Required |
|---------|------|--------|-----------------|
| 7 | CUDA base image | ⚠️ Note | NVIDIA-specific (expected) |
| 41-42 | CUDA library paths | ⚠️ Note | CUDA-specific (expected) |

### `docker/Dockerfile.cpu` - CPU Docker

| Line(s) | Code | Status | Action Required |
|---------|------|--------|-----------------|
| 6 | `python:3.10-slim` | ⚠️ Note | x86_64 only |
| 35-36 | CPU PyTorch | ⚠️ Note | x86_64 only |

---

## Implementation Plan

### Phase 1: Critical Code Fixes (Required for MPS to work)

#### Task 1.1: Guard CUDAGraph Call

- **File:** `dia/model.py`
- **Line:** 701
- **Change:** Wrap in `if self.device.type == "cuda":`

#### Task 1.2: Add MPS Check to torch.compile

- **File:** `dia/model.py`
- **Lines:** 656-660
- **Change:** Skip compilation and warn when device is MPS

#### Task 1.3: Fix CLI Device Default

- **File:** `cli.py`
- **Lines:** 72-74
- **Change:** Add MPS to the default device detection chain

### Phase 2: Quality Improvements

#### Task 2.1: Fix simple-mac.py Example

- **File:** `example/simple-mac.py`
- **Line:** 4
- **Change:** `compute_dtype="float32"`

#### Task 2.2: Guard Triton Config in Benchmark

- **File:** `example/benchmark.py`
- **Lines:** 8-10
- **Change:** Only set Triton config on Linux

#### Task 2.3: Add MPS Seed Management (Optional)

- **Files:** `cli.py`, `app.py`
- **Change:** Add `torch.mps.manual_seed(seed)` when MPS is used

### Phase 3: Docker ARM Support

#### Task 3.1: Create Dockerfile.arm

- **Location:** `docker/Dockerfile.arm`
- **Purpose:** Multi-arch support for ARM64 Linux (CPU-only)

#### Task 3.2: Update docker-compose.yml

- **Change:** Add ARM service definition

#### Task 3.3: Documentation for Native macOS

- **Purpose:** Since Docker cannot access MPS, document native installation

---

## Docker Strategy

### Understanding Docker + MPS Limitations

**Key Insight:** Docker on macOS **cannot** access the MPS GPU backend. This is because:

1. MPS requires direct access to Apple's Metal framework
2. Docker containers are isolated from the host GPU
3. Apple does not provide GPU passthrough for containers

### Recommended Docker Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Support Matrix                     │
├─────────────────┬──────────────┬───────────────────────────┤
│ Platform        │ Architecture │ Docker Support            │
├─────────────────┼──────────────┼───────────────────────────┤
│ Linux + NVIDIA  │ x86_64       │ ✅ Dockerfile.gpu (CUDA)  │
│ Linux CPU       │ x86_64       │ ✅ Dockerfile.cpu         │
│ Linux CPU       │ ARM64        │ 🆕 Dockerfile.arm (new)   │
│ macOS + MPS     │ ARM64        │ ❌ Native only (no Docker)│
│ macOS CPU       │ ARM64        │ ⚠️ Possible but slow      │
└─────────────────┴──────────────┴───────────────────────────┘
```

### Proposed Dockerfile.arm

```dockerfile
# Dockerfile.arm - ARM64 Linux deployment for DIA (CPU-only)
# --------------------------------------------------
# Build: docker buildx build --platform linux/arm64 . -f docker/Dockerfile.arm -t dia-arm
# Run:   docker run --rm -p 7860:7860 dia-arm

FROM --platform=linux/arm64 python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-venv \
    libsndfile1 \
    ffmpeg \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 appuser && \
    mkdir -p /app/outputs /app && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

COPY --chown=appuser:appuser . .

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install PyTorch for ARM64
RUN pip install --upgrade pip && \
    pip install torch torchaudio && \
    pip install --no-cache-dir -e .[dev]

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

ENV GRADIO_SERVER_NAME="0.0.0.0"
EXPOSE 7860

CMD ["python3", "app.py"]
```

---

## Memory & Performance Considerations

### System Requirements (96GB Mac)

| Component | Memory Requirement |
|-----------|-------------------|
| Model weights (float32) | ~6.4 GB |
| DAC audio codec | ~1 GB |
| KV caches + activations | ~2-4 GB |
| OS + overhead | ~2 GB |
| **Total estimated** | **~10-12 GB** |

**Result:** 96GB unified memory is more than sufficient.

### Performance Expectations

| Device | dtype | torch.compile | Expected Speed |
|--------|-------|---------------|----------------|
| CUDA (RTX 4090) | float16 | Yes | ~2.2x realtime |
| CUDA (RTX 4090) | float16 | No | ~1.3x realtime |
| MPS (M-series) | float32 | No | ~0.5-0.8x realtime (estimated) |
| CPU (x86) | float32 | No | ~0.1-0.3x realtime |

**Notes:**

- MPS performance varies significantly by chip (M1 vs M2 vs M3)
- float32 uses more memory and is slower than float16
- No torch.compile on MPS means no JIT optimization

### Optimization Opportunities (Future)

1. **Mixed precision:** Some operations could use float16 on MPS
2. **PyTorch MPS optimizations:** Future PyTorch versions may improve MPS performance
3. **Model quantization:** INT8/INT4 quantization could reduce memory and improve speed

---

## Testing Checklist

After implementing the fixes, verify:

- [ ] `python example/simple-mac.py` runs without errors
- [ ] `python cli.py "[S1] Hello world." --output test.wav` uses MPS by default on Mac
- [ ] `python app.py` launches Gradio UI and generates audio on MPS
- [ ] No warnings/errors about CUDA operations on MPS device
- [ ] Audio quality is acceptable (compare to CUDA output)
- [ ] Memory usage is within expected bounds

---

## Summary of Changes Required

### Critical (Must Fix for MPS to Work)

1. **`dia/model.py:701`** - Guard `cudagraph_mark_step_begin()` with CUDA check
2. **`dia/model.py:656-660`** - Add device check before torch.compile
3. **`cli.py:72-74`** - Add MPS to default device selection

### Important (Should Fix for Quality/Consistency)

1. **`example/simple-mac.py:4`** - Change to `compute_dtype="float32"`
2. **`example/benchmark.py:8-10`** - Guard Triton config with platform check
3. **`docker/Dockerfile.arm`** - Create new ARM64 Dockerfile for Linux ARM

### Optional (Nice to Have)

1. Add MPS seed management (`torch.mps.manual_seed()`)
2. Update README with macOS MPS installation instructions
3. Add MPS-specific benchmark script

---

## References

- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Docker Multi-Architecture Builds](https://docs.docker.com/build/building/multi-platform/)
- [Dia Model Repository](https://github.com/nari-labs/dia)
