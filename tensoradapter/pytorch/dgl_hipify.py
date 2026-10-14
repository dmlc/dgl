# Copyright Advanced Micro Devices, Inc.
#  Licensed under the Apache License Version 2.0

import argparse
import json
import os
import sys

from torch.utils.hipify import hipify_python


def main():
    parser = argparse.ArgumentParser(
        description="Top-level script for HIPifying, filling in most common parameters"
    )
    parser.add_argument(
        "--project-directory",
        type=str,
        help="The root of the project. (default: %(default)s)",
    )

    parser.add_argument(
        "--output-directory",
        type=str,
        default=None,
        help="The Directory to Store the Hipified Project",
        required=False,
    )

    args = parser.parse_args()
    if args.project_directory is not None:
        project_directory = args.project_directory
    if args.output_directory:
        output_directory = args.output_directory
    else:
        output_directory = args.project_directory

    foo = hipify_python.hipify(
        project_directory=project_directory,
        output_directory=output_directory,
        is_pytorch_extension=True,
    )


if __name__ == "__main__":
    main()
