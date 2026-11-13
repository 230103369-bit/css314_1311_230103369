
import csv
import math
import multiprocessing as mp
import multiprocessing.pool
import time
from pathlib import Path

from harness import emit_verify_tag


STUDENT_ID = 230103369

PHYSICAL_CORES = 8
LOGICAL_CORES = 8

STRONG_N = 1_200_000
WEAK_N0 = 200_000

WARMUP = 2
RUNS = 7

OUTPUT_DIR = Path(__file__).resolve().parent / "evidence"


def stage_a(n: int) -> float:
    count = int(n * 0.05)
    total = 0.0

    for i in range(count):
        x = (i % 1000 + 1) * 0.001

        for _ in range(5):
            total += math.sin(x)
            total += math.sqrt(x)

    return total


def stage_b_worker(task: tuple[int, int]) -> float:
    start, end = task
    total = 0.0

    for i in range(start, end):
        x = (i % 1000 + 1) * 0.001
        acc = x

        for _ in range(10):
            acc = acc * 1.000001 + math.sin(x)
            acc += math.sqrt(x)

        total += acc

    return total


def make_partitions(n: int, p: int) -> list[tuple[int, int]]:
    base, remainder = divmod(n, p)

    partitions = []
    start = 0

    for worker_id in range(p):
        size = base + (worker_id < remainder)
        end = start + size

        partitions.append((start, end))
        start = end

    return partitions


def run_trial(pool: mp.pool.Pool, n: int, p: int) -> dict:
    partitions = make_partitions(n, p)

    start_total = time.perf_counter_ns()

    start_a = time.perf_counter_ns()
    result_a = stage_a(n)
    end_a = time.perf_counter_ns()

    results_b = pool.map(stage_b_worker, partitions)

    end_total = time.perf_counter_ns()

    checksum = result_a + sum(results_b)

    if not math.isfinite(checksum):
        raise RuntimeError("Non-finite checksum")

    return {
        "total_ms": (end_total - start_total) / 1e6,
        "stage_a_ms": (end_a - start_a) / 1e6,
        "checksum": checksum,
    }


def benchmark(pool: mp.pool.Pool, n: int, p: int) -> dict:
    for _ in range(WARMUP):
        run_trial(pool, n, p)

    samples = []

    for _ in range(RUNS):
        samples.append(run_trial(pool, n, p))

    samples.sort(key=lambda item: item["total_ms"])

    median = samples[RUNS // 2]

    return {
        "n": n,
        "p": p,
        "min_ms": samples[0]["total_ms"],
        "median_ms": median["total_ms"],
        "iqr_ms": (
            samples[(3 * RUNS) // 4]["total_ms"]
            - samples[RUNS // 4]["total_ms"]
        ),
        "stage_a_ms": median["stage_a_ms"],
        "checksum": median["checksum"],
    }


def save_csv(filename: str, rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def run_strong_scaling() -> list[dict]:
    print("\nPART 1 — STRONG SCALING", flush=True)
    print(f"Fixed N = {STRONG_N:,}", flush=True)

    results = []
    baseline = None
    serial_fraction = None

    for p in range(1, LOGICAL_CORES + 1):
        with mp.Pool(processes=p) as pool:
            result = benchmark(pool, STRONG_N, p)

        if p == 1:
            baseline = result["median_ms"]
            serial_fraction = result["stage_a_ms"] / baseline

        speedup = baseline / result["median_ms"]
        efficiency = speedup / p

        result["speedup"] = speedup
        result["efficiency"] = efficiency
        result["serial_fraction"] = serial_fraction

        results.append(result)

        print(
            f"p={p} "
            f"T={result['median_ms']:.3f} ms "
            f"StageA={result['stage_a_ms']:.3f} ms "
            f"S={speedup:.4f} "
            f"E={efficiency:.4f}",
            flush=True
        )

    amdahl_cap = 1 / serial_fraction

    print(f"\nSerial fraction s = {serial_fraction:.6f}")
    print(f"Amdahl theoretical cap = {amdahl_cap:.4f}")
    print(f"Observed S({LOGICAL_CORES}) = {results[-1]['speedup']:.4f}")

    if PHYSICAL_CORES < LOGICAL_CORES:
        before = results[PHYSICAL_CORES - 1]["speedup"]
        after = results[PHYSICAL_CORES]["speedup"]

        print(
            f"Boundary marginal speedup = {after - before:.4f}"
        )
    else:
        print(
            "Pphys = Plog: no Pphys -> Pphys+1 "
            "transition is available."
        )

        if LOGICAL_CORES >= 5:
            s4 = results[3]["speedup"]
            s5 = results[4]["speedup"]

            print(
                f"Apple M2 p=4 -> p=5 speedup change: "
                f"{s5 - s4:.4f}"
            )

    save_csv("phase2_strong.csv", results)

    return results


def run_weak_scaling() -> list[dict]:
    print("\nPART 2 — WEAK SCALING", flush=True)
    print(f"Per-worker load N0 = {WEAK_N0:,}", flush=True)

    results = []
    baseline = None

    for p in range(1, LOGICAL_CORES + 1):
        n = p * WEAK_N0

        with mp.Pool(processes=p) as pool:
            result = benchmark(pool, n, p)

        if p == 1:
            baseline = result["median_ms"]

        weak_efficiency = baseline / result["median_ms"]

        alpha = (
            result["stage_a_ms"]
            / result["median_ms"]
        )

        gustafson = p - alpha * (p - 1)

        result["weak_efficiency"] = weak_efficiency
        result["alpha"] = alpha
        result["gustafson_speedup"] = gustafson

        results.append(result)

        print(
            f"p={p} "
            f"N={n:,} "
            f"T={result['median_ms']:.3f} ms "
            f"WeakE={weak_efficiency:.4f} "
            f"alpha={alpha:.4f} "
            f"Gustafson={gustafson:.4f}",
            flush=True
        )

    save_csv("phase2_weak.csv", results)

    return results


def main() -> None:
    print("PHASE 2 — STRONG VS WEAK SCALING")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores: {PHYSICAL_CORES}")
    print(f"Logical cores: {LOGICAL_CORES}")
    print(f"Warmups: {WARMUP}")
    print(f"Measured runs: {RUNS}")

    strong = run_strong_scaling()
    weak = run_weak_scaling()

    strong_speedup = strong[-1]["speedup"]
    gustafson_speedup = weak[-1]["gustafson_speedup"]

    print("\nPHASE 2 SUMMARY")
    print(f"Strong speedup at p=8: {strong_speedup:.4f}")
    print(f"Gustafson speedup at p=8: {gustafson_speedup:.4f}")

    print("\nPHASE 2 VERIFICATION")
    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            strong_speedup,
            gustafson_speedup
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
