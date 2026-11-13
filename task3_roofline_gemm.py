
import csv
import ctypes
import math
import multiprocessing as mp
import statistics
import time
from multiprocessing import shared_memory
from pathlib import Path

from harness import emit_verify_tag


STUDENT_ID = 230103369
P = 8
N = 256

L1D_BYTES = 65_536
L2_BYTES = 4_194_304

BUFFER_BYTES = 64_000_000
FLOPS_PER_WORKER = 60_000_000
FMA_ITERATIONS = FLOPS_PER_WORKER // 32

BLOCK_SIZES = (8, 16, 32, 64, 96, 128)

WARMUP = 1
RUNS = 3

OUTPUT_DIR = Path(__file__).resolve().parent / "evidence"


def median_iqr(values):
    ordered = sorted(values)
    n = len(ordered)

    return (
        statistics.median(ordered),
        ordered[(3 * n) // 4] - ordered[n // 4],
    )



def memory_worker(
    src_name,
    dst_name,
    start,
    end,
    start_event,
    ready_queue,
    done_queue,
):
    src = shared_memory.SharedMemory(name=src_name)
    dst = shared_memory.SharedMemory(name=dst_name)

    try:
        source = src.buf[start:end]
        destination = dst.buf[start:end]

        try:
            ready_queue.put(1)

            if not start_event.wait(timeout=120):
                raise TimeoutError("Bandwidth start timeout")

            destination[:] = source
            done_queue.put(1)

        finally:
            source.release()
            destination.release()

    finally:
        src.close()
        dst.close()




def measure_bandwidth():
    print("\nPART 3A — DRAM BANDWIDTH", flush=True)

    a = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_BYTES
    )
    b = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_BYTES
    )

    try:
        ctypes.memset(
            ctypes.addressof(ctypes.c_char.from_buffer(a.buf)),
            0,
            BUFFER_BYTES
        )
        ctypes.memset(
            ctypes.addressof(ctypes.c_char.from_buffer(b.buf)),
            1,
            BUFFER_BYTES
        )

        def trial():
            start_event = mp.Event()
            ready_queue = mp.Queue()
            done_queue = mp.Queue()
            processes = []

            chunk = BUFFER_BYTES // P

            try:
                for worker_id in range(P):
                    start = worker_id * chunk
                    end = (
                        BUFFER_BYTES
                        if worker_id == P - 1
                        else start + chunk
                    )

                    process = mp.Process(
                        target=memory_worker,
                        args=(
                            b.name,
                            a.name,
                            start,
                            end,
                            start_event,
                            ready_queue,
                            done_queue,
                        )
                    )

                    process.start()
                    processes.append(process)

                for _ in range(P):
                    ready_queue.get(timeout=120)

                start_time = time.perf_counter_ns()
                start_event.set()

                for _ in range(P):
                    done_queue.get(timeout=120)

                elapsed = (
                    time.perf_counter_ns() - start_time
                ) / 1e9

                for process in processes:
                    process.join(timeout=120)

                    if process.exitcode != 0:
                        raise RuntimeError(
                            f"Bandwidth worker failed: "
                            f"{process.exitcode}"
                        )

                return elapsed

            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join()

                ready_queue.close()
                done_queue.close()
                ready_queue.join_thread()
                done_queue.join_thread()

        for _ in range(WARMUP):
            trial()

        samples = [
            trial()
            for _ in range(RUNS)
        ]

        median_s, iqr_s = median_iqr(samples)

        bytes_transferred = 2 * BUFFER_BYTES
        bandwidth = bytes_transferred / median_s / 1e9

        print(f"Median: {median_s:.6f} s")
        print(f"IQR: {iqr_s:.6f} s")
        print(f"Bandwidth: {bandwidth:.4f} GB/s")

        return bandwidth, median_s

    finally:
        a.close()
        b.close()
        a.unlink()
        b.unlink()




def compute_worker(worker_id, result_queue):
    x0 = 0.01
    x1 = 0.02
    x2 = 0.03
    x3 = 0.04
    x4 = 0.05
    x5 = 0.06
    x6 = 0.07
    x7 = 0.08
    x8 = 0.09
    x9 = 0.10
    x10 = 0.11
    x11 = 0.12
    x12 = 0.13
    x13 = 0.14
    x14 = 0.15
    x15 = 0.16

    multiplier = 1.00000001
    increment = 0.00000001

    start = time.perf_counter()

    for _ in range(FMA_ITERATIONS):
        x0 = x0 * multiplier + increment
        x1 = x1 * multiplier + increment
        x2 = x2 * multiplier + increment
        x3 = x3 * multiplier + increment
        x4 = x4 * multiplier + increment
        x5 = x5 * multiplier + increment
        x6 = x6 * multiplier + increment
        x7 = x7 * multiplier + increment
        x8 = x8 * multiplier + increment
        x9 = x9 * multiplier + increment
        x10 = x10 * multiplier + increment
        x11 = x11 * multiplier + increment
        x12 = x12 * multiplier + increment
        x13 = x13 * multiplier + increment
        x14 = x14 * multiplier + increment
        x15 = x15 * multiplier + increment

    elapsed = time.perf_counter() - start

    checksum = (
        x0 + x1 + x2 + x3
        + x4 + x5 + x6 + x7
        + x8 + x9 + x10 + x11
        + x12 + x13 + x14 + x15
    )

    result_queue.put(
        (worker_id, elapsed, checksum)
    )


def measure_compute():
    print("\nPART 3A — COMPUTE CEILING", flush=True)

    def trial():
        queue = mp.Queue()
        processes = []

        start = time.perf_counter()

        for worker_id in range(P):
            process = mp.Process(
                target=compute_worker,
                args=(worker_id, queue)
            )
            process.start()
            processes.append(process)

        results = [
            queue.get(timeout=1800)
            for _ in range(P)
        ]

        for process in processes:
            process.join()

        elapsed = time.perf_counter() - start

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    "Compute worker failed"
                )

        queue.close()
        queue.join_thread()

        checksum = sum(item[2] for item in results)

        if not math.isfinite(checksum):
            raise RuntimeError("Invalid compute checksum")

        return elapsed

    for _ in range(WARMUP):
        trial()

    samples = [trial() for _ in range(RUNS)]
    median_s, iqr_s = median_iqr(samples)

    total_flops = FLOPS_PER_WORKER * P
    gflops = total_flops / median_s / 1e9

    print(f"Median: {median_s:.6f} s")
    print(f"IQR: {iqr_s:.6f} s")
    print(f"Compute ceiling: {gflops:.4f} GFLOP/s")

    return gflops, median_s


def gemm_worker(
    a_name,
    b_name,
    c_name,
    row_start,
    row_end,
    variant,
    block_size,
):
    a_shm = shared_memory.SharedMemory(name=a_name)
    b_shm = shared_memory.SharedMemory(name=b_name)
    c_shm = shared_memory.SharedMemory(name=c_name)

    a = b = c = None

    try:
        a = (ctypes.c_double * (N * N)).from_buffer(a_shm.buf)
        b = (ctypes.c_double * (N * N)).from_buffer(b_shm.buf)
        c = (ctypes.c_double * (N * N)).from_buffer(c_shm.buf)

        if variant == "naive":
            for i in range(row_start, row_end):
                for j in range(N):
                    acc = 0.0

                    for k in range(N):
                        acc += a[i * N + k] * b[k * N + j]

                    c[i * N + j] = acc

        elif variant == "reordered":
            for i in range(row_start, row_end):
                for k in range(N):
                    aik = a[i * N + k]
                    b_offset = k * N
                    c_offset = i * N

                    for j in range(N):
                        c[c_offset + j] += aik * b[b_offset + j]

        elif variant == "blocked":
            bs = block_size

            for ii in range(row_start, row_end, bs):
                i_end = min(ii + bs, row_end)

                for jj in range(0, N, bs):
                    j_end = min(jj + bs, N)

                    for kk in range(0, N, bs):
                        k_end = min(kk + bs, N)

                        for i in range(ii, i_end):
                            c_offset = i * N

                            for k in range(kk, k_end):
                                aik = a[i * N + k]
                                b_offset = k * N

                                for j in range(jj, j_end):
                                    c[c_offset + j] += (
                                        aik * b[b_offset + j]
                                    )

        else:
            raise ValueError("Unknown GEMM variant")

    finally:
        del a, b, c
        a_shm.close()
        b_shm.close()
        c_shm.close()


def create_matrix_memory():
    size = N * N * 8

    a = shared_memory.SharedMemory(create=True, size=size)
    b = shared_memory.SharedMemory(create=True, size=size)
    c = shared_memory.SharedMemory(create=True, size=size)

    av = bv = None

    try:
        av = memoryview(a.buf).cast("d")
        bv = memoryview(b.buf).cast("d")

        for i in range(N):
            for j in range(N):
                av[i * N + j] = (
                    ((i * 7 + j * 3) % 101) - 50
                ) / 101.0

                bv[i * N + j] = (
                    ((i * 5 - j * 11) % 97) - 48
                ) / 97.0

    finally:
        if av is not None:
            av.release()
        if bv is not None:
            bv.release()

    return a, b, c


def run_gemm(a, b, c, variant, block_size=0):
    ctypes.memset(
    ctypes.addressof(ctypes.c_char.from_buffer(c.buf)),
    0,
    len(c.buf)
)
    processes = []
    rows_per_worker = N // P

    start = time.perf_counter()

    for worker_id in range(P):
        row_start = worker_id * rows_per_worker
        row_end = (
            N if worker_id == P - 1
            else row_start + rows_per_worker
        )

        process = mp.Process(
            target=gemm_worker,
            args=(
                a.name,
                b.name,
                c.name,
                row_start,
                row_end,
                variant,
                block_size,
            )
        )

        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    elapsed = time.perf_counter() - start

    for process in processes:
        if process.exitcode != 0:
            raise RuntimeError(
                f"GEMM worker failed: {process.exitcode}"
            )

    result = memoryview(c.buf).cast("d")
    try:
        output = list(result)
    finally:
        result.release()

    return elapsed, output


def benchmark_gemm(a, b, c, variant, block_size=0):
    for _ in range(WARMUP):
        run_gemm(a, b, c, variant, block_size)

    times = []
    output = None

    for _ in range(RUNS):
        elapsed, output = run_gemm(
            a, b, c, variant, block_size
        )
        times.append(elapsed)

    median_s, iqr_s = median_iqr(times)

    gflops = (2 * N**3) / median_s / 1e9

    return {
        "variant": variant,
        "block_size": block_size,
        "median_s": median_s,
        "iqr_s": iqr_s,
        "gflops": gflops,
        "output": output,
    }


def max_difference(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def measure_gemm():
    print("\nPART 3B — GEMM", flush=True)

    a, b, c = create_matrix_memory()

    try:
        results = []

        naive = benchmark_gemm(a, b, c, "naive")
        results.append(naive)

        print(
            f"Naive: {naive['median_s']:.4f} s, "
            f"{naive['gflops']:.4f} GFLOP/s",
            flush=True
        )

        reordered = benchmark_gemm(a, b, c, "reordered")
        results.append(reordered)

        print(
            f"Reordered: {reordered['median_s']:.4f} s, "
            f"{reordered['gflops']:.4f} GFLOP/s",
            flush=True
        )

        best_blocked = None

        for bs in BLOCK_SIZES:
            result = benchmark_gemm(
                a, b, c, "blocked", bs
            )

            difference = max_difference(
                result["output"],
                reordered["output"]
            )

            result["max_error"] = difference
            result["tile_bytes"] = 3 * bs * bs * 8
            result["intensity"] = bs / 12

            if difference >= 1e-9:
                raise RuntimeError(
                    f"GEMM correctness failed at BS={bs}: "
                    f"max error={difference}"
                )

            results.append(result)

            if (
                best_blocked is None
                or result["median_s"] < best_blocked["median_s"]
            ):
                best_blocked = result

            print(
                f"BS={bs:3d} "
                f"T={result['median_s']:.4f} s "
                f"GFLOP/s={result['gflops']:.4f} "
                f"Error={difference:.3e} "
                f"Tile={result['tile_bytes']} B "
                f"Intensity={result['intensity']:.4f}",
                flush=True
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        path = OUTPUT_DIR / "phase3_gemm.csv"

        with path.open("w", newline="", encoding="utf-8") as file:
            fields = [
                "variant",
                "block_size",
                "median_s",
                "iqr_s",
                "gflops",
                "max_error",
                "tile_bytes",
                "intensity",
            ]

            writer = csv.DictWriter(
                file, fieldnames=fields
            )
            writer.writeheader()

            for result in results:
                writer.writerow({
                    key: result.get(key, "")
                    for key in fields
                })

        print(f"Saved: {path}")

        return naive, reordered, best_blocked

    finally:
        a.close()
        b.close()
        c.close()

        a.unlink()
        b.unlink()
        c.unlink()


def main():
    print("PHASE 3 — ROOFLINE AND CACHE-BLOCKED GEMM")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Processes: {P}")
    print(f"L1d: {L1D_BYTES} bytes")
    print(f"L2: {L2_BYTES} bytes")
    print(f"Matrix size: {N} x {N}")

    bandwidth, bandwidth_time = measure_bandwidth()
    compute, compute_time = measure_compute()

    ridge_point = compute / bandwidth

    print("\nROOFLINE SUMMARY")
    print(f"Bandwidth: {bandwidth:.4f} GB/s")
    print(f"Compute: {compute:.4f} GFLOP/s")
    print(f"Ridge point: {ridge_point:.4f} FLOP/byte")

    naive, reordered, blocked = measure_gemm()

    print("\nGEMM SUMMARY")
    print(f"Naive GFLOP/s: {naive['gflops']:.4f}")
    print(f"Reordered GFLOP/s: {reordered['gflops']:.4f}")
    print(f"Best block size: {blocked['block_size']}")
    print(f"Blocked GFLOP/s: {blocked['gflops']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "phase3_roofline.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["bandwidth_gbs", bandwidth])
        writer.writerow(["bandwidth_median_s", bandwidth_time])
        writer.writerow(["compute_gflops", compute])
        writer.writerow(["compute_median_s", compute_time])
        writer.writerow(["ridge_flops_per_byte", ridge_point])

    print("\nPHASE 3 VERIFICATION")
    print(
        emit_verify_tag(
            "PHASE3",
            STUDENT_ID,
            naive["gflops"],
            blocked["gflops"]
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
