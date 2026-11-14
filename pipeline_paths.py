import argparse
import os
from pathlib import Path


def resolve_run_folder(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--run-folder",
        type=Path,
        default=os.environ.get("CONSTRUCTION_SAFETY_RUN_FOLDER"),
        help=(
            "Inference run directory. Defaults to "
            "CONSTRUCTION_SAFETY_RUN_FOLDER when set."
        ),
    )
    args = parser.parse_args()

    if args.run_folder is None:
        parser.error(
            "provide --run-folder or set CONSTRUCTION_SAFETY_RUN_FOLDER"
        )

    run_folder = args.run_folder.resolve()

    if not run_folder.is_dir():
        parser.error(f"run folder does not exist: {run_folder}")

    return run_folder
