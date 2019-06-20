import os
import pickle

# Read the cached preprocessed dataset from pickle.
# If the cached file does not exist, preprocess the dataset and cache it.
# Dataset names ending with '-imp' indicates that the task would be implicit feedback
# instead of rating prediction.
# E.g. "movielens10m-imp" indicates to perform implicit feedback (link prediction) on
# MovieLens-10M.
def load_data(args):
    cache_file = args.cache
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            ml = pickle.load(f)
    else:
        if args.dataset in [
                'movielens1m',
                'movielens10m',
                'movielens10m-imp']:
            from rec.datasets.movielens import MovieLens
            ml = MovieLens(args.raw_dataset_path)
        elif args.dataset == 'movielens20m':
            # MovieLens 20M has a different format than prior versions
            from rec.datasets.movielens import MovieLens20M
            ml = MovieLens20M(args.raw_dataset_path)

        with open(cache_file, 'wb') as f:
            pickle.dump(ml, f, protocol=4)

    return ml
