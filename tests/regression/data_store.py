import argparse
import logging
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data store for distributed test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_store", type=str, required=True, help="data store like S3 URI"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        help="target dataset name to fetch",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output directory to save target dataset",
    )
    args, _ = parser.parse_known_args()

    src_dir = os.path.join(args.data_store, args.data_name)
    dst_dir = os.path.join(args.output_dir, args.data_name)
    os.system(f"aws s3 sync {src_dir} {dst_dir}")
    os.system(f"ls -lh {dst_dir}")

    logging.info(f"Finished to download {args.data_name} to {dst_dir}")
