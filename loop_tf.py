#!/usr/bin/env python3
"""Hold GPU memory until interrupted.

Example:
    python scripts/occupy_gpu.py --device 0 --memory-fraction 0.9
"""

import argparse
import signal
import sys
import time

import torch


STOP = False


def handle_stop(signum, frame):
    del signum, frame
    global STOP
    STOP = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Allocate GPU memory and keep the process alive until Ctrl+C.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to occupy.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.9,
        help="Fraction of free GPU memory to allocate when --memory-mib is not set.",
    )
    parser.add_argument(
        "--memory-mib",
        type=int,
        default=None,
        help="Exact amount of GPU memory to allocate in MiB.",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=256,
        help="Allocation chunk size in MiB.",
    )
    parser.add_argument(
        "--busy",
        action="store_true",
        help="Also run small matrix multiplications to keep GPU compute active.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds between status prints.",
    )
    return parser.parse_args()


def bytes_to_mib(value):
    return value / 1024 / 1024


def allocate_memory(device, target_mib, chunk_mib):
    tensors = []
    allocated_mib = 0

    while allocated_mib < target_mib:
        this_chunk_mib = min(chunk_mib, target_mib - allocated_mib)
        num_float32 = int(this_chunk_mib * 1024 * 1024 // 4)

        try:
            tensors.append(torch.empty(num_float32, dtype=torch.float32, device=device))
            tensors[-1].fill_(1.0)
            torch.cuda.synchronize(device)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            print(
                f"Stopped at {allocated_mib:.0f} MiB because CUDA ran out of memory.",
                flush=True,
            )
            break

        allocated_mib += this_chunk_mib
        print(f"Allocated {allocated_mib:.0f} / {target_mib:.0f} MiB", flush=True)

    return tensors


def main():
    args = parse_args()
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment.")

    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise SystemExit(
            f"Invalid --device {args.device}; found {torch.cuda.device_count()} CUDA device(s)."
        )

    if args.memory_mib is None and not 0 < args.memory_fraction <= 1:
        raise SystemExit("--memory-fraction must be in the range (0, 1].")

    if args.memory_mib is not None and args.memory_mib <= 0:
        raise SystemExit("--memory-mib must be positive.")

    if args.chunk_mib <= 0:
        raise SystemExit("--chunk-mib must be positive.")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if args.memory_mib is None:
        target_mib = int(bytes_to_mib(free_bytes) * args.memory_fraction)
    else:
        target_mib = args.memory_mib

    print(
        (
            f"Using {torch.cuda.get_device_name(device)} on cuda:{args.device}; "
            f"free={bytes_to_mib(free_bytes):.0f} MiB, "
            f"total={bytes_to_mib(total_bytes):.0f} MiB, "
            f"target={target_mib:.0f} MiB"
        ),
        flush=True,
    )

    held_tensors = allocate_memory(device, target_mib, args.chunk_mib)
    if not held_tensors:
        raise SystemExit("No GPU memory was allocated.")

    print("Holding GPU memory. Press Ctrl+C to release and exit.", flush=True)

    workload = None
    if args.busy:
        workload = torch.randn((2048, 2048), device=device)

    last_print = 0.0
    while not STOP:
        if args.busy:
            workload = workload @ workload
            workload = workload / workload.norm()
            torch.cuda.synchronize(device)
        else:
            time.sleep(0.2)

        now = time.monotonic()
        if now - last_print >= args.sleep:
            a = torch.randn((20480, 20480), device=device)
            b = torch.randn((20480, 20480), device=device)
            c = a @ b
            print("result is ", c)
            print(
                f"Still holding {bytes_to_mib(torch.cuda.memory_allocated(device)):.0f} MiB.",
                flush=True,
            )
            last_print = now
    print("Releasing GPU memory.", flush=True)
    held_tensors.clear()
    del held_tensors
    if workload is not None:
        del workload
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
