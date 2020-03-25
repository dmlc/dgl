import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", nargs=1, type=str, choices=[
                        'pytorch', 'tensorflow', 'mxnet'], help="Set default backend")
    args = parser.parse_args()
    default_dir = os.path.join(os.path.expanduser('~'), '.dgl')
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)
    config_path = os.path.join(default_dir, '.dgl', 'config.json')
    with open(config_path, "w") as config_file: 
        json.dump({'backend': args.backend[0]}, config_file)
        