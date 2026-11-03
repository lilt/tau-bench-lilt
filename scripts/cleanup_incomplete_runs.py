#!/usr/bin/env python3
"""Delete incomplete simulation runs from the simulations directory."""

import argparse
import sys
from pathlib import Path

from tau2.data_model.simulation import Results
from tau2.utils.utils import DATA_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Delete incomplete simulation runs from the simulations directory."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DATA_DIR / "simulations",
        help="Directory containing simulation files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show details about each file",
    )
    args = parser.parse_args()

    if not args.dir.exists():
        print(f"Directory {args.dir} does not exist")
        sys.exit(1)

    incomplete_files = []
    complete_files = []

    for json_file in sorted(args.dir.glob("*.json")):
        try:
            with open(json_file, "r") as fp:
                results = Results.model_validate_json(fp.read())

            num_tasks = len(results.tasks)
            num_trials = results.info.num_trials
            num_simulations = len(results.simulations)
            expected = num_tasks * num_trials

            if num_simulations < expected:
                incomplete_files.append((json_file, num_simulations, expected))
                if args.verbose:
                    print(
                        f"INCOMPLETE: {json_file.name} ({num_simulations}/{expected} simulations)"
                    )
            else:
                complete_files.append(json_file)
                if args.verbose:
                    print(
                        f"COMPLETE:   {json_file.name} ({num_simulations}/{expected})"
                    )

        except Exception as e:
            print(f"ERROR parsing {json_file.name}: {e}", file=sys.stderr)
            continue

    print(
        f"\nFound {len(incomplete_files)} incomplete and {len(complete_files)} complete runs"
    )

    if not incomplete_files:
        print("Nothing to delete")
        return

    print("\nIncomplete runs to delete:")
    for path, actual, expected in incomplete_files:
        print(f"  {path.name} ({actual}/{expected})")

    if args.dry_run:
        print("\n[DRY RUN] No files deleted")
    else:
        confirm = input("\nDelete these files? (y/N): ").strip().lower()
        if confirm == "y":
            for path, _, _ in incomplete_files:
                path.unlink()
                print(f"Deleted: {path.name}")
            print(f"\nDeleted {len(incomplete_files)} files")
        else:
            print("Aborted")


if __name__ == "__main__":
    main()
