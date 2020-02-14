from utils import load_data

def main(args):
    train_set, val_set, test_set = load_data()

if __name__ == '__main__':
    from argparse import ArgumentParser

    from configure import reaction_center_config

    parser = ArgumentParser(description='Reaction Center Identification')
    args = parser.parse_args().__dict__
    args.update(reaction_center_config)
