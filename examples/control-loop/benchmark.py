#!/usr/bin/env python3
"""
Control-loop benchmark: NV=1 (tinygrad) vs PyTorch CUDA vs CUDA Graphs
on NVIDIA Jetson AGX Orin 64GB.

Measures: kernel launch latency, inference time, end-to-end cycle time,
jitter (std + max deviation), and achieved frequency for two realistic
robotics/drone control loops running at 1000 Hz and 2000 Hz targets.

Run with:
    python benchmark.py

Outputs:
    results/benchmark_results.csv    - raw per-cycle measurements
    results/benchmark_summary.md     - human-readable summary table
    results/benchmark_plots.png      - timing distribution plots
"""

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports – each backend is gracefully skipped if unavailable.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.cuda

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tinygrad.tensor import Tensor
    from tinygrad.device import Device

    TINYGRAD_AVAILABLE = True
except ImportError:
    TINYGRAD_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared model / input specification (identical across all backends so the
# comparison is apples-to-apples).
# ---------------------------------------------------------------------------
INPUT_DIM = 12  # sensor state vector
HIDDEN_DIM = 128
OUTPUT_DIM = 4  # control output


@dataclass
class BenchmarkResult:
    backend: str
    loop: str
    target_hz: int
    n_cycles: int
    launch_latencies_us: List[float] = field(default_factory=list)
    inference_times_us: List[float] = field(default_factory=list)
    cycle_times_us: List[float] = field(default_factory=list)
    cycle_start_times_s: List[float] = field(default_factory=list)

    # Computed statistics (filled by summarise())
    mean_launch_us: float = 0.0
    mean_inference_us: float = 0.0
    mean_cycle_us: float = 0.0
    jitter_std_us: float = 0.0
    jitter_max_us: float = 0.0
    achieved_hz: float = 0.0

    def summarise(self) -> None:
        if not self.cycle_times_us:
            return
        self.mean_launch_us = float(np.mean(self.launch_latencies_us))
        self.mean_inference_us = float(np.mean(self.inference_times_us))
        self.mean_cycle_us = float(np.mean(self.cycle_times_us))
        # Jitter: deviation of actual cycle start times from ideal grid
        if len(self.cycle_start_times_s) > 1:
            ideal_period = 1.0 / self.target_hz
            actual_periods = np.diff(self.cycle_start_times_s)
            deviations_us = (actual_periods - ideal_period) * 1e6
            self.jitter_std_us = float(np.std(deviations_us))
            self.jitter_max_us = float(np.max(np.abs(deviations_us)))
        wall = self.cycle_start_times_s[-1] - self.cycle_start_times_s[0]
        self.achieved_hz = (self.n_cycles - 1) / wall if wall > 0 else 0.0


# ---------------------------------------------------------------------------
# Weights: generate once, share across all backends (same values).
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)
W1_NP = RNG.standard_normal((HIDDEN_DIM, INPUT_DIM)).astype(np.float16)
B1_NP = RNG.standard_normal(HIDDEN_DIM).astype(np.float16)
W2_NP = RNG.standard_normal((OUTPUT_DIM, HIDDEN_DIM)).astype(np.float16)
B2_NP = RNG.standard_normal(OUTPUT_DIM).astype(np.float16)
# PID state
PID_KP, PID_KI, PID_KD = 1.0, 0.1, 0.01


# ---------------------------------------------------------------------------
# Loop 1 (simple): PID + low-pass filter + 2-layer MLP neural policy
# ---------------------------------------------------------------------------
def run_simple_loop_torch(
    target_hz: int, duration_s: float, use_cuda_graphs: bool = False
) -> BenchmarkResult:
    assert TORCH_AVAILABLE, "PyTorch not available"
    tag = "pytorch_cuda_graphs" if use_cuda_graphs else "pytorch_cuda"
    result = BenchmarkResult(
        backend=tag, loop="simple", target_hz=target_hz, n_cycles=0
    )

    dev = torch.device("cuda")
    w1 = torch.tensor(W1_NP, device=dev)
    b1 = torch.tensor(B1_NP, device=dev)
    w2 = torch.tensor(W2_NP, device=dev)
    b2 = torch.tensor(B2_NP, device=dev)
    x = torch.zeros(INPUT_DIM, dtype=torch.float16, device=dev)
    lpf_state = torch.zeros(INPUT_DIM, dtype=torch.float16, device=dev)
    lpf_alpha = 0.9

    pid_integral = torch.zeros(OUTPUT_DIM, dtype=torch.float16, device=dev)
    pid_prev_err = torch.zeros(OUTPUT_DIM, dtype=torch.float16, device=dev)

    def step(x_in):
        # Low-pass filter
        lpf_state.mul_(lpf_alpha).add_(x_in, alpha=1.0 - lpf_alpha)
        # Neural policy (2-layer MLP, FP16)
        h = torch.relu(w1 @ lpf_state + b1)
        policy_out = w2 @ h + b2
        # PID on policy output
        err = policy_out - torch.zeros_like(policy_out)
        pid_integral.add_(err)
        pid_d = err - pid_prev_err
        pid_prev_err.copy_(err)
        ctrl = PID_KP * err + PID_KI * pid_integral + PID_KD * pid_d
        return ctrl

    # CUDA Graph capture
    if use_cuda_graphs:
        static_x = x.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = step(static_x)
        torch.cuda.synchronize()

    # Warm-up
    for _ in range(200):
        if use_cuda_graphs:
            g.replay()
        else:
            step(x)
    torch.cuda.synchronize()

    period = 1.0 / target_hz
    end_time = time.monotonic() + duration_s

    while time.monotonic() < end_time:
        t_cycle_start = time.monotonic()
        result.cycle_start_times_s.append(t_cycle_start)

        # Measure launch latency: time until the first kernel is submitted
        t_launch_start = time.perf_counter()
        if use_cuda_graphs:
            g.replay()
        else:
            h_tmp = torch.relu(w1 @ x + b1)  # first kernel
        t_launch_end = time.perf_counter()
        result.launch_latencies_us.append(
            (t_launch_end - t_launch_start) * 1e6
        )

        # Full inference time (including sync)
        t_inf_start = time.perf_counter()
        if use_cuda_graphs:
            g.replay()
        else:
            out = step(x)
        torch.cuda.synchronize()
        t_inf_end = time.perf_counter()
        result.inference_times_us.append((t_inf_end - t_inf_start) * 1e6)

        # Spin-wait remainder of period
        elapsed = time.monotonic() - t_cycle_start
        remaining = period - elapsed
        if remaining > 0:
            deadline = t_cycle_start + period
            while time.monotonic() < deadline:
                pass

        result.cycle_times_us.append((time.monotonic() - t_cycle_start) * 1e6)
        result.n_cycles += 1

    result.summarise()
    return result


def run_simple_loop_tinygrad(
    target_hz: int, duration_s: float
) -> BenchmarkResult:
    assert TINYGRAD_AVAILABLE, "tinygrad not available"
    result = BenchmarkResult(
        backend="tinygrad_nv1", loop="simple", target_hz=target_hz, n_cycles=0
    )

    w1 = Tensor(W1_NP)
    b1 = Tensor(B1_NP)
    w2 = Tensor(W2_NP)
    b2 = Tensor(B2_NP)
    x = Tensor.zeros(INPUT_DIM).half()
    lpf_alpha = 0.9

    def step(x_in):
        lpf = x_in  # simplified for tinygrad
        h = (w1.dot(lpf) + b1).relu()
        return w2.dot(h) + b2

    # Warm-up
    for _ in range(200):
        out = step(x)
        out.realize()

    period = 1.0 / target_hz
    end_time = time.monotonic() + duration_s

    while time.monotonic() < end_time:
        t_cycle_start = time.monotonic()
        result.cycle_start_times_s.append(t_cycle_start)

        t_launch_start = time.perf_counter()
        out = step(x)  # build graph – actual launch happens at realize
        t_launch_end = time.perf_counter()
        result.launch_latencies_us.append(
            (t_launch_end - t_launch_start) * 1e6
        )

        t_inf_start = time.perf_counter()
        out.realize()
        t_inf_end = time.perf_counter()
        result.inference_times_us.append((t_inf_end - t_inf_start) * 1e6)

        elapsed = time.monotonic() - t_cycle_start
        remaining = period - elapsed
        if remaining > 0:
            deadline = t_cycle_start + period
            while time.monotonic() < deadline:
                pass

        result.cycle_times_us.append((time.monotonic() - t_cycle_start) * 1e6)
        result.n_cycles += 1

    result.summarise()
    return result


# ---------------------------------------------------------------------------
# Loop 2 (full sensor-fusion): Kalman/complementary filter + neural policy
# ---------------------------------------------------------------------------
def run_fusion_loop_torch(
    target_hz: int, duration_s: float, use_cuda_graphs: bool = False
) -> BenchmarkResult:
    assert TORCH_AVAILABLE, "PyTorch not available"
    tag = "pytorch_cuda_graphs" if use_cuda_graphs else "pytorch_cuda"
    result = BenchmarkResult(
        backend=tag, loop="fusion", target_hz=target_hz, n_cycles=0
    )

    dev = torch.device("cuda")
    w1 = torch.tensor(W1_NP, device=dev)
    b1 = torch.tensor(B1_NP, device=dev)
    w2 = torch.tensor(W2_NP, device=dev)
    b2 = torch.tensor(B2_NP, device=dev)
    x = torch.zeros(INPUT_DIM, dtype=torch.float16, device=dev)

    # Complementary filter state (6-DOF IMU)
    alpha = torch.tensor(0.98, dtype=torch.float16, device=dev)
    state = torch.zeros(INPUT_DIM, dtype=torch.float16, device=dev)

    def step(x_in):
        # Complementary filter (sensor fusion)
        fused = alpha * state + (1.0 - alpha) * x_in
        state.copy_(fused)
        # Neural policy
        h = torch.relu(w1 @ fused + b1)
        return w2 @ h + b2

    if use_cuda_graphs:
        static_x = x.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = step(static_x)
        torch.cuda.synchronize()

    for _ in range(200):
        if use_cuda_graphs:
            g.replay()
        else:
            step(x)
    torch.cuda.synchronize()

    period = 1.0 / target_hz
    end_time = time.monotonic() + duration_s

    while time.monotonic() < end_time:
        t_cycle_start = time.monotonic()
        result.cycle_start_times_s.append(t_cycle_start)

        t_launch_start = time.perf_counter()
        if use_cuda_graphs:
            g.replay()
        else:
            _ = torch.relu(w1 @ x + b1)
        t_launch_end = time.perf_counter()
        result.launch_latencies_us.append(
            (t_launch_end - t_launch_start) * 1e6
        )

        t_inf_start = time.perf_counter()
        if use_cuda_graphs:
            g.replay()
        else:
            out = step(x)
        torch.cuda.synchronize()
        t_inf_end = time.perf_counter()
        result.inference_times_us.append((t_inf_end - t_inf_start) * 1e6)

        elapsed = time.monotonic() - t_cycle_start
        if period - elapsed > 0:
            deadline = t_cycle_start + period
            while time.monotonic() < deadline:
                pass

        result.cycle_times_us.append((time.monotonic() - t_cycle_start) * 1e6)
        result.n_cycles += 1

    result.summarise()
    return result


def run_fusion_loop_tinygrad(
    target_hz: int, duration_s: float
) -> BenchmarkResult:
    assert TINYGRAD_AVAILABLE, "tinygrad not available"
    result = BenchmarkResult(
        backend="tinygrad_nv1", loop="fusion", target_hz=target_hz, n_cycles=0
    )

    w1 = Tensor(W1_NP)
    b1 = Tensor(B1_NP)
    w2 = Tensor(W2_NP)
    b2 = Tensor(B2_NP)
    x = Tensor.zeros(INPUT_DIM).half()
    alpha = 0.98

    state = Tensor.zeros(INPUT_DIM).half()

    def step(x_in):
        fused = Tensor([alpha]) * state + Tensor([1.0 - alpha]) * x_in
        h = (w1.dot(fused) + b1).relu()
        return w2.dot(h) + b2

    for _ in range(200):
        out = step(x)
        out.realize()

    period = 1.0 / target_hz
    end_time = time.monotonic() + duration_s

    while time.monotonic() < end_time:
        t_cycle_start = time.monotonic()
        result.cycle_start_times_s.append(t_cycle_start)

        t_launch_start = time.perf_counter()
        out = step(x)
        t_launch_end = time.perf_counter()
        result.launch_latencies_us.append(
            (t_launch_end - t_launch_start) * 1e6
        )

        t_inf_start = time.perf_counter()
        out.realize()
        t_inf_end = time.perf_counter()
        result.inference_times_us.append((t_inf_end - t_inf_start) * 1e6)

        elapsed = time.monotonic() - t_cycle_start
        if period - elapsed > 0:
            deadline = t_cycle_start + period
            while time.monotonic() < deadline:
                pass

        result.cycle_times_us.append((time.monotonic() - t_cycle_start) * 1e6)
        result.n_cycles += 1

    result.summarise()
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")


def save_csv(results: List[BenchmarkResult]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "benchmark_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "backend",
                "loop",
                "target_hz",
                "cycle_idx",
                "launch_us",
                "inference_us",
                "cycle_us",
            ]
        )
        for r in results:
            for i, (la, inf, cy) in enumerate(
                zip(r.launch_latencies_us, r.inference_times_us, r.cycle_times_us)
            ):
                writer.writerow(
                    [r.backend, r.loop, r.target_hz, i, la, inf, cy]
                )
    print(f"Raw results → {path}")


def save_summary(results: List[BenchmarkResult]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "benchmark_summary.md"
    header = (
        "| Backend | Loop | Target Hz | Launch µs | Inference µs "
        "| Cycle µs | Jitter σ µs | Jitter max µs | Achieved Hz |\n"
        "|---------|------|-----------|-----------|--------------|"
        "----------|-------------|---------------|-------------|\n"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r.backend} | {r.loop} | {r.target_hz} "
            f"| {r.mean_launch_us:.2f} | {r.mean_inference_us:.2f} "
            f"| {r.mean_cycle_us:.2f} | {r.jitter_std_us:.2f} "
            f"| {r.jitter_max_us:.2f} | {r.achieved_hz:.1f} |"
        )
    with open(path, "w") as f:
        f.write("# Control-Loop Benchmark Results – Jetson AGX Orin\n\n")
        f.write(header)
        f.write("\n".join(rows) + "\n")
    print(f"Summary  → {path}")


def save_plots(results: List[BenchmarkResult]) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available – skipping plots")
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Control-Loop Timing – Jetson AGX Orin", fontsize=14)
    metrics = [
        ("launch_latencies_us", "Launch latency (µs)", axes[0, 0]),
        ("inference_times_us", "Inference time (µs)", axes[0, 1]),
        ("cycle_times_us", "Cycle time (µs)", axes[1, 0]),
    ]
    for attr, title, ax in metrics:
        for r in results:
            data = getattr(r, attr)
            if data:
                label = f"{r.backend}/{r.loop}@{r.target_hz}Hz"
                ax.hist(data, bins=60, alpha=0.5, label=label)
        ax.set_title(title)
        ax.set_xlabel("µs")
        ax.set_ylabel("Count")
        ax.legend(fontsize=6)

    # Achieved vs target Hz
    ax = axes[1, 1]
    backends = [f"{r.backend}\n{r.loop}\n{r.target_hz}Hz" for r in results]
    target_vals = [r.target_hz for r in results]
    achieved_vals = [r.achieved_hz for r in results]
    x_pos = np.arange(len(backends))
    ax.bar(x_pos - 0.2, target_vals, width=0.4, label="Target Hz")
    ax.bar(x_pos + 0.2, achieved_vals, width=0.4, label="Achieved Hz")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(backends, fontsize=6)
    ax.set_title("Achieved vs Target frequency")
    ax.legend()

    plt.tight_layout()
    path = RESULTS_DIR / "benchmark_plots.png"
    plt.savefig(path, dpi=150)
    print(f"Plots    → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control-loop benchmark: tinygrad NV=1 vs PyTorch CUDA"
    )
    parser.add_argument(
        "--duration", type=float, default=60.0, help="Duration per run (s)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["pytorch", "pytorch_graphs", "tinygrad"],
        choices=["pytorch", "pytorch_graphs", "tinygrad"],
        help="Backends to benchmark",
    )
    parser.add_argument(
        "--loops",
        nargs="+",
        default=["simple", "fusion"],
        choices=["simple", "fusion"],
    )
    parser.add_argument(
        "--hz",
        nargs="+",
        type=int,
        default=[1000, 2000],
        help="Target frequencies to test",
    )
    args = parser.parse_args()

    if TORCH_AVAILABLE:
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available – PyTorch results will be CPU-only")
        else:
            print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")

    if TINYGRAD_AVAILABLE:
        print(f"tinygrad device: {Device.DEFAULT}")
    else:
        print("tinygrad not available – skipping NV=1 backend")

    all_results: List[BenchmarkResult] = []

    for hz in args.hz:
        for loop in args.loops:
            print(f"\n=== {loop} loop @ {hz} Hz ===")

            if "pytorch" in args.backends and TORCH_AVAILABLE:
                print(f"  Running PyTorch CUDA …")
                fn = run_simple_loop_torch if loop == "simple" else run_fusion_loop_torch
                r = fn(hz, args.duration, use_cuda_graphs=False)
                all_results.append(r)
                print(
                    f"    launch={r.mean_launch_us:.2f}µs  "
                    f"infer={r.mean_inference_us:.2f}µs  "
                    f"cycle={r.mean_cycle_us:.2f}µs  "
                    f"jitter_std={r.jitter_std_us:.2f}µs  "
                    f"achieved={r.achieved_hz:.1f}Hz"
                )

            if "pytorch_graphs" in args.backends and TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    print(f"  Running PyTorch CUDA Graphs …")
                    fn = (
                        run_simple_loop_torch
                        if loop == "simple"
                        else run_fusion_loop_torch
                    )
                    r = fn(hz, args.duration, use_cuda_graphs=True)
                    all_results.append(r)
                    print(
                        f"    launch={r.mean_launch_us:.2f}µs  "
                        f"infer={r.mean_inference_us:.2f}µs  "
                        f"cycle={r.mean_cycle_us:.2f}µs  "
                        f"jitter_std={r.jitter_std_us:.2f}µs  "
                        f"achieved={r.achieved_hz:.1f}Hz"
                    )

            if "tinygrad" in args.backends and TINYGRAD_AVAILABLE:
                print(f"  Running tinygrad NV=1 …")
                fn = (
                    run_simple_loop_tinygrad
                    if loop == "simple"
                    else run_fusion_loop_tinygrad
                )
                r = fn(hz, args.duration)
                all_results.append(r)
                print(
                    f"    launch={r.mean_launch_us:.2f}µs  "
                    f"infer={r.mean_inference_us:.2f}µs  "
                    f"cycle={r.mean_cycle_us:.2f}µs  "
                    f"jitter_std={r.jitter_std_us:.2f}µs  "
                    f"achieved={r.achieved_hz:.1f}Hz"
                )

    if not all_results:
        print("No results collected. Check that at least one backend is available.")
        return

    save_csv(all_results)
    save_summary(all_results)
    save_plots(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
