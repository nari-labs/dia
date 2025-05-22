import argparse
import time
import os
import numpy as np
import torch
import statistics
from pathlib import Path
from typing import List, Dict, Tuple

from dia.model import Dia, ComputeDtype


def verify_cuda_setup() -> bool:
    """Verify CUDA is available and properly configured."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available on this system.")
        return False
    
    # Print CUDA device information
    device_count = torch.cuda.device_count()
    print(f"CUDA Information:")
    print(f"- CUDA Available: {torch.cuda.is_available()}")
    print(f"- Number of CUDA Devices: {device_count}")
    
    for i in range(device_count):
        device = torch.cuda.device(i)
        props = torch.cuda.get_device_properties(device)
        print(f"- Device {i}: {props.name}")
        print(f"  - CUDA Capability: {props.major}.{props.minor}")
        print(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
    
    return True


def run_benchmark(
    model: Dia,
    texts: List[str],
    max_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    num_runs: int,
    warm_up: bool = True,
    detailed_timing: bool = False,
) -> Dict:
    """Run benchmark for the model on CUDA.
    
    Args:
        model: The Dia model instance
        texts: List of texts to generate audio from
        max_tokens: Maximum number of tokens to generate
        cfg_scale: Classifier-free guidance scale
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        num_runs: Number of benchmark runs
        warm_up: Whether to do a warm-up run (not counted in results)
        detailed_timing: Whether to collect detailed timing per token
        
    Returns:
        Dictionary of performance metrics
    """
    # Verify model is on CUDA
    if not next(model.model.parameters()).is_cuda:
        print("WARNING: Model does not appear to be on CUDA!")
    
    # Optionally do a warm-up run to ensure any initialization happens before benchmarking
    if warm_up:
        print("Warming up GPU...")
        _ = model.generate(
            texts[0],
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=45
        )
    
    # Benchmark metrics
    generation_times = []
    tokens_per_second = []
    memory_usage = []
    
    for i in range(num_runs):
        # Rotate through the texts if we have more runs than texts
        text = texts[i % len(texts)]
        
        # Clear CUDA cache before each run
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated() / (1024**2)  # Convert to MB
        
        # Time the generation
        start_time = time.time()
        output = model.generate(
            text,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=45
        )
        end_time = time.time()
        
        # Calculate metrics
        elapsed = end_time - start_time
        tokens_sec = max_tokens / elapsed
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
        memory_used = peak_memory - start_memory
        
        # Store results
        generation_times.append(elapsed)
        tokens_per_second.append(tokens_sec)
        memory_usage.append(memory_used)
        
        # Print run information
        print(f"Run {i+1}/{num_runs}:")
        print(f"  - Time: {elapsed:.2f} seconds")
        print(f"  - Speed: {tokens_sec:.2f} tokens/second")
        print(f"  - Memory: {memory_used:.2f} MB")
    
    # Calculate aggregate metrics
    results = {
        "times": generation_times,
        "avg_time": statistics.mean(generation_times),
        "median_time": statistics.median(generation_times),
        "min_time": min(generation_times),
        "max_time": max(generation_times),
        "tokens_per_second": statistics.mean(tokens_per_second),
        "avg_memory_mb": statistics.mean(memory_usage),
    }
    
    return results


def print_gpu_results(results: Dict, compute_dtype: str) -> None:
    """Print detailed GPU benchmark results."""
    print("\n" + "="*60)
    print(f"GPU BENCHMARK RESULTS (Precision: {compute_dtype})")
    print("="*60)
    
    print(f"Performance Metrics:")
    print(f"- Average Generation Time: {results['avg_time']:.2f} seconds")
    print(f"- Median Generation Time: {results['median_time']:.2f} seconds")
    print(f"- Best Time: {results['min_time']:.2f} seconds")
    print(f"- Worst Time: {results['max_time']:.2f} seconds")
    print(f"- Average Generation Speed: {results['tokens_per_second']:.2f} tokens/second")
    print(f"- Average Memory Usage: {results['avg_memory_mb']:.2f} MB")
    
    if len(results['times']) > 1:
        std_dev = statistics.stdev(results['times'])
        variance_pct = (std_dev / results['avg_time']) * 100
        print(f"- Run Consistency: {variance_pct:.2f}% variance")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Dia model performance on CUDA")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--max-tokens", type=int, default=860, help="Maximum tokens to generate")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--temperature", type=float, default=1.3, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling value")
    parser.add_argument("--compare-precision", action="store_true", help="Compare FP16 vs BF16 precision")
    
    args = parser.parse_args()
    
    # Verify CUDA setup
    if not verify_cuda_setup():
        print("CUDA verification failed. Cannot continue with GPU benchmarking.")
        return
    
    # Test texts to use for benchmarking
    benchmark_texts = [
        "[S1] This is a test of the generation speed. [S2] The quick brown fox jumps over the lazy dog.",
        "[S1] Let's benchmark this model to see how it performs. [S2] Performance testing is important.",
        "[S1] CUDA should be faster than CPU for neural networks. [S2] But by how much? Let's find out."
    ]
    
    # CUDA device
    cuda_device = torch.device("cuda")
    
    # If comparing precision modes
    if args.compare_precision and torch.cuda.is_bf16_supported():
        # Float16 benchmarking
        print("\nRunning GPU benchmark with FP16 precision...")
        fp16_model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=cuda_device)
        fp16_results = run_benchmark(
            model=fp16_model,
            texts=benchmark_texts,
            max_tokens=args.max_tokens,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_p=args.top_p,
            num_runs=args.num_runs
        )
        del fp16_model
        torch.cuda.empty_cache()
        
        # BFloat16 benchmarking
        print("\nRunning GPU benchmark with BF16 precision...")
        bf16_model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="bfloat16", device=cuda_device)
        bf16_results = run_benchmark(
            model=bf16_model,
            texts=benchmark_texts,
            max_tokens=args.max_tokens,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_p=args.top_p,
            num_runs=args.num_runs
        )
        del bf16_model
        torch.cuda.empty_cache()
        
        # Print results
        print_gpu_results(fp16_results, "FP16")
        print_gpu_results(bf16_results, "BF16")
        
        # Compare the two
        speed_diff = (bf16_results['tokens_per_second'] / fp16_results['tokens_per_second'] - 1) * 100
        time_diff = (fp16_results['avg_time'] / bf16_results['avg_time'] - 1) * 100
        
        print("\n" + "="*60)
        print("PRECISION COMPARISON")
        print("="*60)
        print(f"BF16 vs FP16 Speed Difference: {speed_diff:.2f}% ({'faster' if speed_diff > 0 else 'slower'})")
        print(f"BF16 vs FP16 Time Difference: {time_diff:.2f}% ({'faster' if time_diff > 0 else 'slower'})")
        print("="*60)
    
    else:
        # Standard FP16 benchmarking
        print("\nRunning GPU benchmark with FP16 precision...")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=cuda_device)
        results = run_benchmark(
            model=model,
            texts=benchmark_texts,
            max_tokens=args.max_tokens,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_p=args.top_p,
            num_runs=args.num_runs
        )
        print_gpu_results(results, "FP16")
        
        if args.compare_precision and not torch.cuda.is_bf16_supported():
            print("\nWarning: BF16 comparison requested but your GPU does not support BF16.")


if __name__ == "__main__":
    main() 