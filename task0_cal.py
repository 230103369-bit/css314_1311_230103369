
import multiprocessing as mp
import time

from harness import emit_verify_tag


STUDENT_ID = 230103369
CALLS = 1_000_000


def main():
    logical_cores = mp.cpu_count()

    start = time.perf_counter_ns()

    for _ in range(CALLS):
        time.perf_counter_ns()

    elapsed_ns = time.perf_counter_ns() - start
    average_latency_ns = elapsed_ns / CALLS

    print("PHASE 0 — HARDWARE CALIBRATION")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical CPUs: {logical_cores}")
    print(f"Timer calls: {CALLS:,}")
    print(f"Total elapsed: {elapsed_ns:,} ns")
    print(f"Average timer latency: {average_latency_ns:.4f} ns")

    print(
        emit_verify_tag(
            "PHASE0",
            STUDENT_ID,
            float(logical_cores),
            average_latency_ns
        )
    )


if __name__ == "__main__":
    main()
