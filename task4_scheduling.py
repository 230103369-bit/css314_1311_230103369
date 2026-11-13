
import csv
import math
import multiprocessing as mp
import statistics
import time
from pathlib import Path

from harness import emit_verify_tag


STUDENT_ID = 230103369

N = 8000
P = 8

CHUNK_SIZES = (1, 8, 64, 512)

WARMUP = 1
RUNS = 5

OUTPUT_DIR = Path(__file__).resolve().parent / "evidence"


def workload(i: int) -> float:
    k = 1 + (i * i) // 1600
    x = (i % 100 + 1) * 0.001
    result = 0.0

    for j in range(k):
        result += math.sin(x + j * 0.00001)

    return result


def static_block_worker(worker_id, active_times, results):
    start_index = worker_id * N // P
    end_index = (worker_id + 1) * N // P

    start = time.perf_counter_ns()
    total = 0.0

    for i in range(start_index, end_index):
        total += workload(i)

    elapsed = (time.perf_counter_ns() - start) / 1e9

    active_times[worker_id] = elapsed
    results[worker_id] = total


def static_cyclic_worker(worker_id, active_times, results):
    start = time.perf_counter_ns()
    total = 0.0

    for i in range(worker_id, N, P):
        total += workload(i)

    elapsed = (time.perf_counter_ns() - start) / 1e9

    active_times[worker_id] = elapsed
    results[worker_id] = total


def dynamic_worker(
    worker_id,
    next_index,
    chunk_size,
    active_times,
    results,
):
    start = time.perf_counter_ns()
    total = 0.0

    while True:
        with next_index.get_lock():
            chunk_start = next_index.value
            next_index.value += chunk_size

        if chunk_start >= N:
            break

        chunk_end = min(chunk_start + chunk_size, N)

        for i in range(chunk_start, chunk_end):
            total += workload(i)

    elapsed = (time.perf_counter_ns() - start) / 1e9

    active_times[worker_id] = elapsed
    results[worker_id] = total


def run_trial(strategy: str, chunk_size: int = 0) -> dict:
    active_times = mp.Array("d", P)
    results = mp.Array("d", P)

    next_index = mp.Value("i", 0)

    processes = []

    start_wall = time.perf_counter_ns()

    try:
        for worker_id in range(P):
            if strategy == "static_block":
                target = static_block_worker
                args = (
                    worker_id,
                    active_times,
                    results,
                )

            elif strategy == "static_cyclic":
                target = static_cyclic_worker
                args = (
                    worker_id,
                    active_times,
                    results,
                )

            elif strategy == "dynamic":
                target = dynamic_worker
                args = (
                    worker_id,
                    next_index,
                    chunk_size,
                    active_times,
                    results,
                )

            else:
                raise ValueError(
                    f"Unknown strategy: {strategy}"
                )

            process = mp.Process(
                target=target,
                args=args,
            )

            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        wall_s = (
            time.perf_counter_ns() - start_wall
        ) / 1e9

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: {process.exitcode}"
                )

        times = list(active_times)
        checksum = sum(results)

        if not math.isfinite(checksum):
            raise RuntimeError("Invalid checksum")

        mean_active = statistics.mean(times)

        imbalance = (
            max(times) / mean_active
            if mean_active > 0
            else float("nan")
        )

        return {
            "wall_s": wall_s,
            "active_times": times,
            "imbalance": imbalance,
            "checksum": checksum,
        }

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


def benchmark(strategy: str, chunk_size: int = 0) -> dict:
    for _ in range(WARMUP):
        run_trial(strategy, chunk_size)

    samples = [
        run_trial(strategy, chunk_size)
        for _ in range(RUNS)
    ]

    samples.sort(key=lambda x: x["wall_s"])

    median_result = samples[RUNS // 2]

    wall_times = [
        sample["wall_s"]
        for sample in samples
    ]

    median_wall = statistics.median(wall_times)

    q1 = wall_times[RUNS // 4]
    q3 = wall_times[(3 * RUNS) // 4]

    return {
        "strategy": strategy,
        "chunk_size": chunk_size,
        "median_ms": median_wall * 1000,
        "min_ms": min(wall_times) * 1000,
        "iqr_ms": (q3 - q1) * 1000,
        "imbalance": median_result["imbalance"],
        "checksum": median_result["checksum"],
        "active_times": median_result["active_times"],
    }


def save_csv(filename: str, rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / filename

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def main():
    print("PHASE 4 — SCHEDULING AND LOAD IMBALANCE")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Elements: {N:,}")
    print(f"Processes: {P}")
    print(f"Warmups: {WARMUP}")
    print(f"Measured runs: {RUNS}")

    rows = []
    detailed_rows = []

    configurations = [
        ("static_block", 0),
        ("static_cyclic", 0),
        ("dynamic", 1),
        ("dynamic", 8),
        ("dynamic", 64),
        ("dynamic", 512),
    ]

    print("\nSCHEDULING BENCHMARK", flush=True)

    baseline_checksum = None
    static_block_result = None
    best_dynamic = None

    for strategy, chunk_size in configurations:
        result = benchmark(strategy, chunk_size)

        if baseline_checksum is None:
            baseline_checksum = result["checksum"]

        difference = abs(
            result["checksum"] - baseline_checksum
        )

        tolerance = max(
            1e-6,
            abs(baseline_checksum) * 1e-10,
        )

        if difference > tolerance:
            raise RuntimeError(
                f"Checksum mismatch: {strategy}, "
                f"chunk={chunk_size}, diff={difference}"
            )

        if strategy == "static_block":
            static_block_result = result

        if strategy == "dynamic":
            if (
                best_dynamic is None
                or result["median_ms"]
                < best_dynamic["median_ms"]
            ):
                best_dynamic = result

        rows.append({
            "strategy": strategy,
            "chunk_size": chunk_size,
            "median_ms": result["median_ms"],
            "min_ms": result["min_ms"],
            "iqr_ms": result["iqr_ms"],
            "imbalance": result["imbalance"],
            "checksum": result["checksum"],
        })

        for worker_id, active_s in enumerate(
            result["active_times"]
        ):
            detailed_rows.append({
                "strategy": strategy,
                "chunk_size": chunk_size,
                "worker_id": worker_id,
                "active_s": active_s,
            })

        label = (
            strategy
            if strategy != "dynamic"
            else f"dynamic C={chunk_size}"
        )

        print(
            f"{label:20s} "
            f"T={result['median_ms']:.3f} ms "
            f"IQR={result['iqr_ms']:.3f} ms "
            f"lambda={result['imbalance']:.4f}",
            flush=True
        )

    save_csv("phase4_scheduling.csv", rows)
    save_csv("phase4_worker_times.csv", detailed_rows)

    improvement = (
        static_block_result["median_ms"]
        / best_dynamic["median_ms"]
    )

    print("\nPHASE 4 SUMMARY")
    print(
        f"Static block: "
        f"{static_block_result['median_ms']:.3f} ms"
    )
    print(
        f"Best dynamic chunk: "
        f"{best_dynamic['chunk_size']}"
    )
    print(
        f"Best dynamic: "
        f"{best_dynamic['median_ms']:.3f} ms"
    )
    print(
        f"Dynamic speedup over static block: "
        f"{improvement:.4f}"
    )

    print("\nPHASE 4 VERIFICATION")
    print(
        emit_verify_tag(
            "PHASE4",
            STUDENT_ID,
            static_block_result["median_ms"],
            best_dynamic["median_ms"],
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
