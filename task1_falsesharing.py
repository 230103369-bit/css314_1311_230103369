
import ctypes
import csv
import multiprocessing as mp
import time
from multiprocessing import shared_memory
from pathlib import Path

from harness import emit_verify_tag


STUDENT_ID = 230103369
PHYSICAL_CORES = 8
LOGICAL_CORES = 8

M = 16_000_000
SHM_SIZE = 4096
STRIDES = (1, 2, 4, 8, 16, 32)

WARMUP = 2
RUNS = 7

MASK = 0xFFFFFFFFFFFFFFFF
MULTIPLIER = 6364136223846793005

OUTPUT_DIR = Path(__file__).resolve().parent / "evidence"


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

        try:
            acc = arr[slot_idx]

            for i in range(iters):
                acc = (acc * MULTIPLIER + (i | 1)) & MASK
                arr[slot_idx] = acc

        finally:
            del arr

    finally:
        shm.close()


def run_trial(p: int, stride: int) -> float:
    if p * stride > 512:
        raise ValueError("Worker slots exceed SharedMemory capacity")

    shm = shared_memory.SharedMemory(
        create=True,
        size=SHM_SIZE
    )

    processes = []

    try:

        base_iters, remainder = divmod(M, p)

        start = time.perf_counter_ns()

        for worker_id in range(p):
            iters = base_iters + (worker_id < remainder)

            process = mp.Process(
                target=fs_worker,
                args=(
                    shm.name,
                    worker_id * stride,
                    iters
                )
            )

            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        elapsed_ms = (
            time.perf_counter_ns() - start
        ) / 1_000_000

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {process.exitcode}"
                )

        return elapsed_ms

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

        shm.close()
        shm.unlink()


def benchmark(p: int, stride: int) -> dict:
    for _ in range(WARMUP):
        run_trial(p, stride)

    samples = [
        run_trial(p, stride)
        for _ in range(RUNS)
    ]

    samples.sort()

    minimum = samples[0]
    median = samples[RUNS // 2]
    iqr = (
        samples[(3 * RUNS) // 4]
        - samples[RUNS // 4]
    )

    throughput = M / (median * 1000)

    return {
        "p": p,
        "stride": stride,
        "offset_bytes": stride * 8,
        "min_ms": minimum,
        "median_ms": median,
        "iqr_ms": iqr,
        "throughput_mops": throughput,
    }


def scaling_metrics(t1: float, tp: float, p: int):
    speedup = t1 / tp
    efficiency = speedup / p

    if p == 1:
        return speedup, efficiency, None

    karp_flatt = (
        (1 / speedup - 1 / p)
        / (1 - 1 / p)
    )

    return speedup, efficiency, karp_flatt


def save_csv(filename: str, rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    path = OUTPUT_DIR / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def main() -> None:
    print("PHASE 1 — FALSE SHARING")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores: {PHYSICAL_CORES}")
    print(f"Logical cores: {LOGICAL_CORES}")
    print(f"Total operations: {M:,}")
    print(f"Warmups: {WARMUP}, measured runs: {RUNS}")

    print("\nPART 1 — STRIDE SWEEP")

    stride_rows = []
    stride_results = {}

    for stride in STRIDES:
        result = benchmark(PHYSICAL_CORES, stride)

        stride_rows.append(result)
        stride_results[stride] = result

        print(
            f"stride={stride:2d} "
            f"offset={stride * 8:3d} B "
            f"median={result['median_ms']:.3f} ms "
            f"IQR={result['iqr_ms']:.3f} ms "
            f"throughput={result['throughput_mops']:.3f} Mops/s",
            flush=True
        )

    save_csv("phase1_stride.csv", stride_rows)

    print("\nPART 2 — PROCESS SCALING")

    scaling_rows = []

    for stride in (1, 16):
        print(f"\nSTRIDE = {stride}")

        baseline = None

        for p in (1, 2, 4, 8):
            if p == PHYSICAL_CORES:
                result = stride_results[stride]
            else:
                result = benchmark(p, stride)

            if p == 1:
                baseline = result["median_ms"]

            speedup, efficiency, e = scaling_metrics(
                baseline,
                result["median_ms"],
                p
            )

            row = {
                "stride": stride,
                "p": p,
                "median_ms": result["median_ms"],
                "speedup": speedup,
                "efficiency": efficiency,
                "karp_flatt": "" if e is None else e,
            }

            scaling_rows.append(row)

            e_text = "N/A" if e is None else f"{e:.4f}"

            print(
                f"p={p:2d} "
                f"T={result['median_ms']:.3f} ms "
                f"S={speedup:.4f} "
                f"E={efficiency:.4f} "
                f"e={e_text}",
                flush=True
            )

    save_csv("phase1_scaling.csv", scaling_rows)

    m1 = stride_results[1]["median_ms"]
    m2 = stride_results[16]["median_ms"]

    print("\nPHASE 1 VERIFICATION")

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            m1,
            m2
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
