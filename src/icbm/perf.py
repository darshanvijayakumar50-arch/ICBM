import argparse
import logging
import sys
from pathlib import Path
import subprocess
import time


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

    return parser.parse_args()


def process_file(input_path: Path, output_path: Path) -> None:
    """TODO: Load file and mesure perf"""

    start = time.perf_counter()

    result = subprocess.run(
        [".venv/bin/python.exe", "main.py", "data/easy/imu.csv", "data/easy/estimated.csv"],
        capture_output=True,
        text=True,
        stdout=None,
        stderr=None,
    )

    elapsed = time.perf_counter() - start

    logger.info(f"Elapsed time: {elapsed:.6f} seconds")
    logger.info(f"Exit code: {result.returncode}")

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

        process_file(args.input, args.output)

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