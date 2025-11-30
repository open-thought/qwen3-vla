#!/usr/bin/env python3
"""
Benchmark inter-GPU communication speed.

Tests NCCL all-reduce performance which is the main communication
primitive used in DDP for gradient synchronization.

Usage:
    # Test with 2 GPUs
    torchrun --nproc_per_node=2 dev/benchmark_gpu_communication.py

    # Test with all available GPUs
    torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) dev/benchmark_gpu_communication.py
"""

import os
import time
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed environment."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}")
        )

    return rank, world_size, local_rank


def check_p2p_access():
    """Check P2P access between GPUs."""
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print("P2P Access Matrix")
    print(f"{'='*60}")

    # Header
    print(f"{'':>8}", end="")
    for j in range(num_gpus):
        print(f"GPU{j:>4}", end="")
    print()

    for i in range(num_gpus):
        print(f"GPU{i:>4}  ", end="")
        for j in range(num_gpus):
            if i == j:
                print(f"{'--':>4}", end="")
            else:
                can_access = torch.cuda.can_device_access_peer(i, j)
                print(f"{'Yes':>4}" if can_access else f"{'No':>4}", end="")
        print()


def benchmark_all_reduce(tensor_size_mb: float, num_iterations: int = 100, warmup: int = 10):
    """Benchmark all-reduce operation."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Create tensor
    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
    tensor = torch.randn(num_elements, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Benchmark
    dist.barrier()
    start = time.perf_counter()

    for _ in range(num_iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    torch.cuda.synchronize()
    dist.barrier()
    end = time.perf_counter()

    elapsed = end - start
    avg_time_ms = (elapsed / num_iterations) * 1000

    # Calculate bandwidth
    # All-reduce sends data to all GPUs and receives from all
    # Total data moved = 2 * (world_size - 1) / world_size * tensor_size (ring algorithm)
    # Simplified: ~2 * tensor_size for bandwidth calculation
    data_gb = tensor_size_mb / 1024
    bandwidth_gbps = (2 * data_gb) / (avg_time_ms / 1000)

    return avg_time_ms, bandwidth_gbps


def benchmark_broadcast(tensor_size_mb: float, num_iterations: int = 100, warmup: int = 10):
    """Benchmark broadcast operation."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)
    tensor = torch.randn(num_elements, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        dist.broadcast(tensor, src=0)
    torch.cuda.synchronize()

    # Benchmark
    dist.barrier()
    start = time.perf_counter()

    for _ in range(num_iterations):
        dist.broadcast(tensor, src=0)

    torch.cuda.synchronize()
    dist.barrier()
    end = time.perf_counter()

    elapsed = end - start
    avg_time_ms = (elapsed / num_iterations) * 1000
    data_gb = tensor_size_mb / 1024
    bandwidth_gbps = data_gb / (avg_time_ms / 1000)

    return avg_time_ms, bandwidth_gbps


def estimate_ddp_overhead(model_size_gb: float, all_reduce_bandwidth_gbps: float):
    """Estimate DDP gradient sync overhead for a model."""
    # DDP syncs gradients which are same size as parameters (for float32)
    # With mixed precision (bf16), gradients are often still float32
    gradient_size_gb = model_size_gb

    # Time to sync gradients
    sync_time_ms = (2 * gradient_size_gb / all_reduce_bandwidth_gbps) * 1000

    return sync_time_ms


def main():
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"\n{'='*60}")
        print("GPU Communication Benchmark")
        print(f"{'='*60}")
        print(f"World size: {world_size} GPUs")
        print(f"CUDA devices: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

        # Check P2P access
        check_p2p_access()

        # Check NCCL environment
        print(f"\nNCCL Environment:")
        for key in ["NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL", "NCCL_SHM_DISABLE", "NCCL_NET_GDR_LEVEL"]:
            val = os.environ.get(key, "not set")
            print(f"  {key}: {val}")

    if world_size < 2:
        if rank == 0:
            print("\nNeed at least 2 GPUs for communication benchmark.")
            print("Run with: torchrun --nproc_per_node=2 dev/benchmark_gpu_communication.py")
        return

    dist.barrier()

    # Benchmark different tensor sizes
    test_sizes_mb = [1, 10, 50, 100, 500, 1000]  # MB

    if rank == 0:
        print(f"\n{'='*60}")
        print("All-Reduce Benchmark (main DDP primitive)")
        print(f"{'='*60}")
        print(f"{'Size (MB)':>12} {'Time (ms)':>12} {'Bandwidth (GB/s)':>18}")
        print("-" * 44)

    all_reduce_results = []
    for size_mb in test_sizes_mb:
        avg_time, bandwidth = benchmark_all_reduce(size_mb, num_iterations=50)
        all_reduce_results.append((size_mb, avg_time, bandwidth))
        if rank == 0:
            print(f"{size_mb:>12} {avg_time:>12.2f} {bandwidth:>18.2f}")

    if rank == 0:
        print(f"\n{'='*60}")
        print("Broadcast Benchmark")
        print(f"{'='*60}")
        print(f"{'Size (MB)':>12} {'Time (ms)':>12} {'Bandwidth (GB/s)':>18}")
        print("-" * 44)

    for size_mb in test_sizes_mb:
        avg_time, bandwidth = benchmark_broadcast(size_mb, num_iterations=50)
        if rank == 0:
            print(f"{size_mb:>12} {avg_time:>12.2f} {bandwidth:>18.2f}")

    # Estimate DDP overhead for common model sizes
    if rank == 0:
        # Use the bandwidth from largest tensor (most representative)
        _, _, peak_bandwidth = all_reduce_results[-1]

        print(f"\n{'='*60}")
        print("Estimated DDP Gradient Sync Overhead")
        print(f"(Based on peak all-reduce bandwidth: {peak_bandwidth:.2f} GB/s)")
        print(f"{'='*60}")
        print(f"{'Model':>20} {'Params':>12} {'Size (GB)':>12} {'Sync Time (ms)':>15}")
        print("-" * 62)

        models = [
            ("Qwen3-VL-2B", 2e9),
            ("Qwen3-VL-4B", 4e9),
            ("Qwen3-VL-7B", 7e9),
            ("LLaMA-8B", 8e9),
            ("LLaMA-70B", 70e9),
        ]

        for name, params in models:
            # bf16 parameters = 2 bytes each, but gradients often float32 = 4 bytes
            grad_size_gb = params * 4 / 1e9
            sync_time = estimate_ddp_overhead(grad_size_gb, peak_bandwidth)
            print(f"{name:>20} {params/1e9:>10.1f}B {grad_size_gb:>12.1f} {sync_time:>15.1f}")

        print(f"\n{'='*60}")
        print("Analysis")
        print(f"{'='*60}")

        # Check if P2P is disabled
        p2p_available = torch.cuda.can_device_access_peer(0, 1) if torch.cuda.device_count() > 1 else False

        if not p2p_available:
            print("⚠ P2P access NOT available between GPUs!")
            print("  Communication goes through CPU/PCIe, which is slower.")
            print("  Typical PCIe 4.0 x16: ~25 GB/s (unidirectional)")
            print("  Typical NVLink: ~300-600 GB/s (bidirectional)")
            print()
            print("  Options to improve:")
            print("  1. Use FSDP with sharding to reduce communication")
            print("  2. Increase batch size to amortize sync overhead")
            print("  3. Use gradient accumulation to reduce sync frequency")
        else:
            print("✓ P2P access available between GPUs")

        # Calculate overhead percentage for Qwen3-VL-4B
        qwen4b_sync_ms = estimate_ddp_overhead(4 * 4, peak_bandwidth)  # 4B params * 4 bytes
        typical_forward_backward_ms = 3000  # ~3s for forward+backward
        overhead_pct = (qwen4b_sync_ms / typical_forward_backward_ms) * 100

        print(f"\nFor Qwen3-VL-4B with ~3s forward+backward:")
        print(f"  Gradient sync overhead: ~{qwen4b_sync_ms:.0f}ms ({overhead_pct:.1f}%)")

        if overhead_pct > 20:
            print(f"  ⚠ High overhead! Consider FSDP or gradient accumulation.")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
