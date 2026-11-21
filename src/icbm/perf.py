import argparse
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import psutil

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False) -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process an input file and write the result to an output file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input file",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Path to the output file",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Show only warnings and errors",
    )

    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Show only logs, without graph",
    )

    parser.add_argument(
        "--interval",
        type=int,
        help="Interval between RAM polls",
        default=100,
    )

    return parser.parse_args()

def benchmark(input_path: Path, output_path: Path, interval_ms:int = 100):
    process = subprocess.Popen(
        [
            sys.executable,
            "src/main4.py",
            input_path,
            output_path,
        ],
        stdout=None,
        stderr=None,
    )

    def monitor_process_exit():
        process.wait()
        stop_event.set()

    threading.Thread(
        target=monitor_process_exit,
        daemon=True
    ).start()

    monitored = psutil.Process(process.pid)

    start = time.perf_counter()
    performance = []  # (elapsed_time, ram_mb)

    stop_event = threading.Event()
    start_time = time.perf_counter()

    peak_ram = 0

    while process.poll() is None:
        try:
            processes = [monitored] + monitored.children(recursive=True)

            ram = sum(
                p.memory_info().rss
                for p in processes
                if p.is_running()
            ) / (1024 ** 2)

            peak_ram = max(peak_ram, ram)

            performance.append((
                time.perf_counter() - start_time,
                ram
            ))

        except psutil.NoSuchProcess:
            pass

        # Wait for the interval, but stop immediately if signalled.
        if stop_event.wait(interval_ms / 1000):
            break

    elapsed = time.perf_counter() - start

    return {
        "exit_code": process.returncode,
        "elapsed_seconds": elapsed,
        "performance": performance,
        "peak_ram_mb": peak_ram,
    }

def plot(performance):

    times, ram_values = zip(*performance)

    plt.figure(figsize=(10, 5))
    plt.plot(times, ram_values, label="RAM Usage")

    plt.xlabel("Time (seconds)")
    plt.ylabel("RAM Usage (MB)")
    plt.title("Process RAM Usage Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main() -> int:
    args = parse_args()

    log_level = (
        logging.WARNING if args.quiet
        else logging.DEBUG if args.verbose
        else logging.INFO
    )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    try:
        if not args.input.is_file():
            logger.error("Input file does not exist: %s", args.input)
            return 1

        logger.info("Input:  %s", args.input)
        logger.info("Output: %s", args.output)

        rc, elapsed, perf, peak = benchmark(args.input, args.output, args.interval).values()

        logger.info("Exit code: %d", rc)
        logger.info("Elapsed time: %.3f seconds", elapsed)
        logger.info("Peak RAM: %.2f MB", peak)

        if not args.no_graph:
            plot(perf)

        logger.info("Processing completed successfully")
        return 0

    except PermissionError:
        logger.exception("Permission denied while accessing files")
        return 1

    except OSError:
        logger.exception("File operation failed")
        return 1

    except Exception:
        logger.exception("Unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())