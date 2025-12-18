# MPS (Apple Silicon) Performance Optimization Research

This document presents in-depth research on MPS-specific optimizations for PyTorch on Apple Silicon, exploring how to extract maximum performance from the Metal Performance Shaders backend.

## Executive Summary

While CUDA benefits from technologies like CUDA Graphs, Triton kernels, and Tensor Cores, MPS has its own optimization ecosystem. This research identifies **actionable optimizations** that could improve Dia TTS performance on Apple Silicon.

### Key Optimization Opportunities

| Optimization | Potential Speedup | Complexity | Applicability to Dia |
|-------------|------------------|------------|---------------------|
| Metal FlashAttention | 2-10x for attention | High | High - attention-heavy model |
| Core ML + ANE Conversion | 10-30x | High | Medium - requires model conversion |
| MLX Framework | 1.5-3x | Medium | Medium - requires rewrite |
| Mixed Precision (autocast) | 1.2-2x | Low | High - easy to implement |
| torch.compile (PyTorch 2.8+) | 1.3-2x | Low | Medium - experimental |
| Batch Size Tuning | 1.1-1.5x | Low | High - easy to test |
| Memory Format (channels_last) | 1.1-1.3x | Low | Low - mainly for CNNs |

---

## 1. Metal FlashAttention

### Overview

[Metal FlashAttention](https://github.com/philipturner/metal-flash-attention) is an open-source port of FlashAttention specifically optimized for Apple Silicon GPUs. It provides dramatic speedups for transformer attention operations.

### Performance Claims

- **43-120% faster** image generation in Stable Diffusion benchmarks
- **Up to 80x speedup** in multi-head attention vs naive MPS implementation
- Orders of magnitude improvement in some benchmarks where standard MPS couldn't complete in reasonable time

### How It Works

From [Draw Things Engineering Blog](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c):

> Metal FlashAttention leverages the `simdgroup_async_copy` API (since A14), an undocumented hardware feature that overlaps compute and load instructions. Inspired by the FlashAttention project, Metal FlashAttention aimed to improve both latency and memory footprint.

### Key Features

1. **Optimized Forward Pass** - Several optimizations decrease total computations and increase numerical stability
2. **Block-Sparse Algorithm** - Automatically detects sparsity in attention matrices
3. **JIT Compilation** - Everything compiled at runtime for optimal performance
4. **Lower Memory Usage** - Backward pass uses less memory than original FlashAttention

### Implementations

- **Swift**: [github.com/philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
- **C++**: Part of [ccv library](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa)

### Applicability to Dia

Dia uses custom attention in `dia/layers.py:139-189` (`custom_scaled_dot_product_attention`). Integrating Metal FlashAttention could significantly speed up the decoder's cross-attention and self-attention operations.

**Challenge**: Would require custom Metal shader integration or using a wrapper library.

---

## 2. Core ML + Apple Neural Engine (ANE)

### Overview

The [Apple Neural Engine (ANE)](https://machinelearning.apple.com/research/neural-engine-transformers) is a dedicated ML accelerator in Apple Silicon that can provide 10x+ speedups over GPU inference for supported operations.

### Performance Numbers

From [Apple's research](https://machinelearning.apple.com/research/neural-engine-transformers):

> The popular Hugging Face distilbert model is **up to 10 times faster** and consumes **14 times less memory** after optimizations for the Apple Neural Engine.

Real-world benchmarks:
- ~4.4ms per inference with CoreML on ANE vs ~44ms on CPU PyTorch (**10x faster**)
- Up to **31x speedup** in some profiling tests
- 8B parameter Llama model: ~33 tokens/sec on M1 Max via Core ML

### ANE Specifications

| Chip | ANE Cores | ANE TOPS (INT8) | ANE TFLOPS (FP16) |
|------|-----------|-----------------|-------------------|
| M1/M1 Pro/M1 Max | 16 | 11 | ~5.5 |
| M2/M2 Pro/M2 Max | 16 | 15.8 | ~8 |
| M3/M3 Pro/M3 Max | 16 | 18 | ~9 |
| M4 | 16 | 38 | ~19 |

### Conversion Workflow

From [coremltools documentation](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html):

```python
import torch
import coremltools as ct

# 1. Set model to eval mode
model.eval()

# 2. Trace the model
example_input = torch.randn(1, 512)
traced_model = torch.jit.trace(model, example_input)

# 3. Convert to Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + ANE
    compute_precision=ct.precision.FLOAT16  # ANE only supports FP16
)

# 4. Save
mlmodel.save("model.mlpackage")
```

### Compute Unit Options

From [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/backends-coreml.html):

| Option | Description |
|--------|-------------|
| `ComputeUnit.ALL` | Uses CPU, GPU, and ANE (default) |
| `ComputeUnit.CPU_ONLY` | CPU only |
| `ComputeUnit.CPU_AND_GPU` | CPU and GPU, no ANE |
| `ComputeUnit.CPU_AND_NE` | CPU and ANE, no GPU |

### ANE Limitations

1. **FP16 only** - ANE only supports half-precision
2. **Operation support** - Not all PyTorch ops have ANE implementations
3. **Dynamic shapes** - Limited support for variable input sizes
4. **Model conversion** - Requires conversion step, not drop-in replacement

### Applicability to Dia

Converting Dia to Core ML could provide significant speedups, especially for:
- Encoder forward pass (fixed-size after text encoding)
- Decoder step (if shapes can be made static)

**Challenge**: Dia's autoregressive generation with dynamic KV caches may be difficult to convert efficiently.

---

## 3. MLX Framework

### Overview

[MLX](https://github.com/ml-explore/mlx) is Apple's own ML framework designed specifically for Apple Silicon, offering better performance than PyTorch MPS in many scenarios.

### Key Advantages

From [MLX documentation](https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html):

> The most striking advantage of using Apple Silicon is the elimination of data transfers between the CPU and GPU. This might sound like a small detail, but in the real world of machine learning projects, these data transfers are a notorious source of latency.

### Performance Comparison

From [academic research (arxiv.org/pdf/2501.14925)](https://arxiv.org/pdf/2501.14925):

| Framework | Relative Performance | Notes |
|-----------|---------------------|-------|
| PyTorch MPS | Baseline | Good compatibility |
| MLX | 1.5-3x faster | Apple-optimized |
| CUDA (RTX 4090) | 3x faster than M2 Ultra | Still fastest overall |

### Unified Memory Model

```python
import mlx.core as mx

# Arrays live in shared memory - no explicit transfers needed
x = mx.array([1, 2, 3])  # Can be used on CPU or GPU seamlessly
```

### Applicability to Dia

Rewriting Dia in MLX could provide significant performance gains, but would require:
- Complete model reimplementation
- Porting DAC codec to MLX
- Maintaining two codebases

**Recommendation**: Consider for a future "Dia-MLX" variant rather than the main codebase.

---

## 4. Mixed Precision with torch.autocast

### Overview

PyTorch MPS now supports [automatic mixed precision (AMP)](https://docs.pytorch.org/docs/stable/amp.html), which can speed up computation while reducing memory usage.

### Supported Data Types

| Data Type | MPS Support | macOS Requirement |
|-----------|-------------|-------------------|
| float32 | ✅ Full | Any |
| float16 | ✅ Full | Any |
| bfloat16 | ✅ Full | macOS Sonoma+ |

### Usage

From [PyTorch documentation](https://docs.pytorch.org/docs/stable/amp.html):

```python
import torch

# Using autocast for inference
with torch.autocast(device_type="mps", dtype=torch.float16):
    output = model(input)

# Using autocast for training with gradient scaling
scaler = torch.amp.GradScaler("mps")
for input, target in data:
    optimizer.zero_grad()
    with torch.autocast(device_type="mps", dtype=torch.float16):
        output = model(input)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### bfloat16 Benefits

From [WWDC23](https://developer.apple.com/videos/play/wwdc2023/10050/):

> Starting with macOS Sonoma, MPSGraph adds support for bfloat16. bfloat16 has the same dynamic range as float32 (8 exponent bits) but reduced precision (7 mantissa bits), making it more numerically stable than float16 for training.

### Performance Notes

From research:
- NVIDIA GPUs with Tensor Cores: **2-4x speedup** with FP16
- Apple Silicon MPS: **0-30% speedup** with FP16 (no dedicated Tensor Cores)
- MLX: **20-30% speedup** with FP16

### Applicability to Dia

Current Dia implementation uses float32 for MPS stability. Testing with autocast could provide modest speedups:

```python
# Potential optimization for Dia
with torch.autocast(device_type="mps", dtype=torch.bfloat16):
    output = model.generate(text, use_torch_compile=False)
```

**Recommendation**: Test bfloat16 autocast on macOS Sonoma+ for potential 10-30% speedup.

---

## 5. torch.compile for MPS (PyTorch 2.8+)

### Current Status

From [PyTorch GitHub Issue #150121](https://github.com/pytorch/pytorch/issues/150121):

> torch.compile support for MPS device is an early prototype and attempt to use it to accelerate end-to-end network is likely to fail.

However, [PyTorch 2.8 (August 2025)](https://michaelbommarito.com/wiki/programming/languages/python/libraries/pytorch-2-8-release/) introduced first official M-series compilation support.

### Usage (PyTorch 2.8+)

```python
device = torch.device("mps")
model = model.to(device)
compiled_model = torch.compile(model, backend="inductor")
```

### Known Limitations

1. FlexAttention not implemented for MPS
2. Dynamic shapes have issues
3. Some reduction operations slower than eager mode
4. May not work for all model architectures

### Applicability to Dia

Our current implementation skips torch.compile on MPS. Once PyTorch 2.8+ is stable:

```python
# Future potential optimization
if use_torch_compile and self.device.type == "mps":
    # Use default mode instead of max-autotune
    self._decoder_step = torch.compile(self._decoder_step, backend="inductor")
```

**Recommendation**: Monitor PyTorch MPS torch.compile development. Test with PyTorch 2.8+ when available.

---

## 6. MPSGraph Operator Fusion

### How It Works

From [WWDC22](https://developer.apple.com/videos/play/wwdc2022/10063/):

> The MPSGraph compiler recognizes all adjacent operations and passes it to the Metal compiler. The Metal compiler fuses the operations together to create a single optimized Metal shader. This leads to no memory overhead and improves performance.

### Stitching Optimization

> For any math operations around hand-tuned MPS kernels, like convolution, matrix multiplication, or reduction, MPSGraph recognizes adjacent stitchable operations to create a region and passes them to the Metal compiler for them to be fused directly into the hand-tuned MPS kernel.

### Performance Impact

> Using the stitching optimization makes GeLU go almost **10 to 50 times faster**.

### Applicability to Dia

PyTorch MPS backend automatically uses MPSGraph, but ensuring operations are graph-compatible helps:

1. **Avoid CPU fallbacks** - Check for `PYTORCH_ENABLE_MPS_FALLBACK` warnings
2. **Use standard operations** - Stick to well-supported ops
3. **Minimize synchronization** - Avoid `torch.mps.synchronize()` in hot paths

---

## 7. Memory Optimization Strategies

### Unified Memory Architecture

From [Apple documentation](https://developer.apple.com/metal/pytorch/):

> Every Apple silicon Mac has a unified memory architecture, providing the GPU with direct access to the full memory store. This makes Mac a great platform for machine learning, enabling users to train larger networks or batch sizes locally.

### Current Limitation

From [PyTorch Issue #140787](https://github.com/pytorch/pytorch/issues/140787):

> "Since Mac with Apple Silicon has unified memory, why in PyTorch we still need to copy tensors from CPU and MPS? This will double the memory usage."

The current MPS implementation doesn't fully leverage unified memory - tensors are still copied between CPU and GPU contexts.

### Memory Management Best Practices

```python
# Clear MPS cache when needed
torch.mps.empty_cache()

# Synchronize before measuring memory
torch.mps.synchronize()

# Check current memory allocation
print(torch.mps.current_allocated_memory())
print(torch.mps.driver_allocated_memory())
```

### Buffer Size Limits

From [Medium article](https://medium.com/@rakshekaraj/optimizing-pytorch-mps-attention-memory-efficient-large-sequence-processing-without-accuracy-5239f565f07b):

> Apple's MPS backend enforces stricter buffer size limits due to the way Metal handles memory allocations. When working with long input sequences (seq_length > 12000), PyTorch on MPS attempts to allocate a single large buffer for the attention computation, which often exceeds Metal's maximum supported buffer size.

### Applicability to Dia

Dia generates sequences up to 3072 tokens, within MPS limits. Optimizations:

1. **Reduce intermediate tensors** - Reuse buffers where possible
2. **Use gradient checkpointing** - If training (not applicable for inference)
3. **Clear cache between generations** - `torch.mps.empty_cache()`

---

## 8. Profiling MPS Performance

### MPS Profiler API

From [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.mps.profiler.profile.html):

```python
import torch.mps.profiler

# Start profiling
torch.mps.profiler.start(mode="interval", wait_until_completed=True)

# Run your model
output = model(input)

# Stop and save trace
torch.mps.profiler.stop()
```

### Metal System Trace

From [Apple Developer documentation](https://developer.apple.com/metal/pytorch/):

> You can visualize the profiling data in Metal System Trace, which is part of Instruments. To use the profiler, call the start method on the MPS profiler package to enable tracing.

### Identifying Bottlenecks

1. **Enable OS Signposts** in Metal System Trace
2. Look for:
   - Long-running operations
   - CPU-GPU synchronization points
   - Memory copy operations
   - Fallback to CPU operations

### Applicability to Dia

Profile Dia to identify optimization targets:

```python
import torch.mps.profiler

torch.mps.profiler.start(mode="interval", wait_until_completed=True)
output = model.generate("[S1] Test text.", use_torch_compile=False)
torch.mps.profiler.stop()
# Analyze in Xcode Instruments
```

---

## 9. Batch Size and Throughput Optimization

### Finding Optimal Batch Size

From [HuggingFace Accelerate docs](https://huggingface.co/docs/accelerate/usage_guides/mps):

> Experiment with different batch sizes to find the optimal batch size for your model and dataset. A larger batch size can increase the parallelism of the computation, but it also requires more memory.

### Batch Size Guidelines

| Memory | Recommended Batch Size | Notes |
|--------|----------------------|-------|
| 8GB | 1-2 | Conservative |
| 16GB | 2-4 | Standard |
| 32GB+ | 4-8 | Can experiment higher |
| 64GB+ | 8-16 | Memory not limiting |

### Gradient Accumulation (Training)

If batch size is limited by memory, use gradient accumulation:

```python
accumulation_steps = 4
for i, (input, target) in enumerate(dataloader):
    output = model(input)
    loss = loss_fn(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Applicability to Dia

Dia currently processes one text input at a time. Potential optimizations:

1. **Batch multiple generations** - Generate multiple outputs in parallel
2. **Optimize for single-item batches** - Current default, ensure no overhead

---

## 10. Apple Silicon GPU Architecture

### Performance Comparison

From [academic research](https://arxiv.org/pdf/2501.14925):

| Chip | GPU Cores | FP32 TFLOPS | Memory BW | Unified Memory |
|------|-----------|-------------|-----------|----------------|
| M1 | 8 | 2.6 | 68 GB/s | Up to 16GB |
| M1 Pro | 16 | 5.2 | 200 GB/s | Up to 32GB |
| M1 Max | 32 | 10.4 | 400 GB/s | Up to 64GB |
| M2 | 10 | 3.6 | 100 GB/s | Up to 24GB |
| M2 Max | 38 | 13.6 | 400 GB/s | Up to 96GB |
| M3 | 10 | 4.1 | 100 GB/s | Up to 24GB |
| M4 | 10 | 4.3 | 120 GB/s | Up to 32GB |

For comparison:
- RTX 4090: 82.6 FP32 TFLOPS, 1008 GB/s bandwidth

### Key Architectural Differences

1. **No Tensor Cores** - Apple GPUs lack dedicated matrix multiplication hardware
2. **Unified Memory** - CPU and GPU share memory pool
3. **Power Efficiency** - 200+ GFLOPS/Watt vs ~50 GFLOPS/Watt for desktop GPUs
4. **No FP64 Support** - Must emulate double precision

---

## Recommendations for Dia TTS

### Short-Term (Easy Wins)

1. **Test bfloat16 autocast** on macOS Sonoma+
   ```python
   with torch.autocast(device_type="mps", dtype=torch.bfloat16):
       output = model.generate(text)
   ```

2. **Profile with MPS Profiler** to identify bottlenecks

3. **Clear MPS cache** between generations
   ```python
   torch.mps.empty_cache()
   ```

4. **Test batch generation** if use case supports it

### Medium-Term (Moderate Effort)

5. **Integrate Metal FlashAttention** for attention operations
   - Would require custom Metal shader integration
   - Potential 2-10x speedup for attention layers

6. **Test torch.compile** with PyTorch 2.8+
   - Use default backend instead of max-autotune
   - Monitor stability and performance

7. **Optimize memory format** for any CNN components

### Long-Term (Significant Effort)

8. **Core ML Conversion** for ANE acceleration
   - Convert encoder to Core ML
   - Potential 10x+ speedup for encoder pass
   - Challenge: Dynamic decoder with KV cache

9. **MLX Port** for maximum Apple Silicon performance
   - Complete rewrite required
   - Would be a separate "Dia-MLX" project

10. **Custom Metal Kernels** for critical operations
    - Requires Metal shader expertise
    - Maximum control over performance

---

## References

### Official Documentation
- [Apple Metal PyTorch Guide](https://developer.apple.com/metal/pytorch/)
- [PyTorch MPS Backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [torch.mps Documentation](https://docs.pytorch.org/docs/stable/mps.html)
- [Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html)

### Research & Technical Articles
- [Profiling Apple Silicon Performance for ML Training (arxiv.org)](https://arxiv.org/pdf/2501.14925)
- [Deploying Transformers on ANE (Apple ML Research)](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Metal FlashAttention 2.0 (Draw Things Engineering)](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)
- [MPS vs MLX Comparison (Medium)](https://medium.com/@koypish/mps-or-mlx-for-domestic-ai-the-answer-will-surprise-you-df4b111de8a0)

### GitHub Repositories
- [Metal FlashAttention](https://github.com/philipturner/metal-flash-attention)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [ANE Transformers](https://github.com/apple/ml-ane-transformers)
- [torch.compile MPS Tracker (#150121)](https://github.com/pytorch/pytorch/issues/150121)

### WWDC Sessions
- [WWDC22: Accelerate ML with Metal](https://developer.apple.com/videos/play/wwdc2022/10063/)
- [WWDC23: Optimize ML for Metal Apps](https://developer.apple.com/videos/play/wwdc2023/10050/)
- [WWDC24: Accelerate ML with Metal](https://developer.apple.com/videos/play/wwdc2024/10218/)

---

*Research compiled: December 2024*
*For Dia TTS MPS Support Implementation*
