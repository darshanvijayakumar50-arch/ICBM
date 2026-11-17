
import argparse
import logging
import sys
from pathlib import Path

import jsbsim

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Command handlers
# --------------------------------------------------

def cmd_list(args):
    """
    Lists available flights
    """
    root = Path(jsbsim.get_default_root_dir())

    print("JSBSim data directory:", root)

    aircraft_dir = root / "aircraft"
    scripts_dir = root / "scripts"

    print("\nAvailable aircraft:")
    for path in aircraft_dir.iterdir():
        if path.is_dir():
            print("-", path.name)

    print("\nAvailable flight scripts:")
    for path in scripts_dir.glob("*.xml"):
        print("-", path.name)

def cmd_run(args):
    print("STUB: TODO")
    


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="mytool",
        description="A Git-style CLI application.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="mytool 1.0.0",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress informational messages.",
    )



    # Subcommands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="<command>",
        required=True,
    )

    # --------------------------------------------------
    # list
    # --------------------------------------------------

    list_parser = subparsers.add_parser(
        "list",
        help="list availabe models",
    )

    list_parser.set_defaults(func=cmd_list)

    # --------------------------------------------------
    # run
    # --------------------------------------------------

    run_parser = subparsers.add_parser(
        "run",
        help="Run the simulation",
    )

    # Positional arguments
    run_parser.add_argument(
        "input",
        type=Path,
        help="Input file path.",
    )

    # Optional arguments
    run_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output.txt"),
        help="Output file path.",
    )


    run_parser.set_defaults(func=cmd_run)
    return parser



def setup_logging(args: argparse.Namespace) -> None:
    """Configure application logging."""

    if args.quiet:
        level = logging.ERROR
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args)

    try:
        return args.func(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Operation interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
