"""
Script that reads from raw MovieLens-1M data and dumps into a pickle
file the following:

* A heterogeneous graph with categorical features.
* A list with all the movie titles.  The movie titles correspond to
  the movie nodes in the heterogeneous graph.

This script exemplifies how to prepare tabular data with textual
features.  Since DGL graphs do not store variable-length features, we
instead put variable-length features into a more suitable container
(e.g. torchtext to handle list of texts)
"""

import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
import torchtext
from builder import PandasGraphBuilder
from data_utils import *

import dgl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("out_directory", type=str)
    args = parser.parse_args()
    directory = args.directory
    out_directory = args.out_directory
    os.makedirs(out_directory, exist_ok=True)

    ## Build heterogeneous graph

    # Load data
    users = []
    with open(os.path.join(directory, "users.dat"), encoding="latin1") as f:
        for l in f:
            id_, gender, age, occupation, zip_ = l.strip().split("::")
            users.append(
                {
                    "user_id": int(id_),
                    "gender": gender,
                    "age": age,
                    "occupation": occupation,
                    "zip": zip_,
                }
            )
    users = pd.DataFrame(users).astype("category")

    movies = []
    with open(os.path.join(directory, "movies.dat"), encoding="latin1") as f:
        for l in f:
            id_, title, genres = l.strip().split("::")
            genres_set = set(genres.split("|"))

            # extract year
            assert re.match(r".*\([0-9]{4}\)$", title)
            year = title[-5:-1]
            title = title[:-6].strip()

            data = {"movie_id": int(id_), "title": title, "year": year}
            for g in genres_set:
                data[g] = True
            movies.append(data)
    movies = pd.DataFrame(movies).astype({"year": "category"})

    ratings = []
    with open(os.path.join(directory, "ratings.dat"), encoding="latin1") as f:
        for l in f:
            user_id, movie_id, rating, timestamp = [
                int(_) for _ in l.split("::")
            ]
            ratings.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
    ratings = pd.DataFrame(ratings)

    # Filter the users and items that never appear in the rating table.
    distinct_users_in_ratings = ratings["user_id"].unique()
    distinct_movies_in_ratings = ratings["movie_id"].unique()
    users = users[users["user_id"].isin(distinct_users_in_ratings)]
    movies = movies[movies["movie_id"].isin(distinct_movies_in_ratings)]

    # Group the movie features into genres (a vector), year (a category), title (a string)
    genre_columns = movies.columns.drop(["movie_id", "title", "year"])
    movies[genre_columns] = movies[genre_columns].fillna(False).astype("bool")
    movies_categorical = movies.drop("title", axis=1)

    # Build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, "user_id", "user")
    graph_builder.add_entities(movies_categorical, "movie_id", "movie")
    graph_builder.add_binary_relations(
        ratings, "user_id", "movie_id", "watched"
    )
    graph_builder.add_binary_relations(
        ratings, "movie_id", "user_id", "watched-by"
    )

    g = graph_builder.build()

    # Assign features.
    # Note that variable-sized features such as texts or images are handled elsewhere.
    for data_type in ["gender", "age", "occupation", "zip"]:
        g.nodes["user"].data[data_type] = torch.LongTensor(
            np.array(users[data_type].cat.codes.values)
        )

    g.nodes["movie"].data["year"] = torch.LongTensor(
        np.array(movies["year"].cat.codes.values)
    )
    g.nodes["movie"].data["genre"] = torch.FloatTensor(
        np.array(movies[genre_columns].values)
    )

    for edge_type in ["watched", "watched-by"]:
        for data_type in ["rating", "timestamp"]:
            g.edges[edge_type].data[data_type] = torch.LongTensor(
                np.array(ratings[data_type].values)
            )

    # Train-validation-test split
    # This is a little bit tricky as we want to select the last interaction for test, and the
    # second-to-last interaction for validation.
    train_indices, val_indices, test_indices = train_test_split_by_time(
        ratings, "timestamp", "user_id"
    )

    # Build the graph with training interactions only.
    train_g = build_train_graph(
        g, train_indices, "user", "movie", "watched", "watched-by"
    )
    assert train_g.out_degrees(etype="watched").min() > 0

    # Build the user-item sparse matrix for validation and test set.
    val_matrix, test_matrix = build_val_test_matrix(
        g, val_indices, test_indices, "user", "movie", "watched"
    )

    ## Build title set

    movie_textual_dataset = {"title": movies["title"].values}

    # The model should build their own vocabulary and process the texts.  Here is one example
    # of using torchtext to pad and numericalize a batch of strings.
    #     field = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
    #     examples = [torchtext.data.Example.fromlist([t], [('title', title_field)]) for t in texts]
    #     titleset = torchtext.data.Dataset(examples, [('title', title_field)])
    #     field.build_vocab(titleset.title, vectors='fasttext.simple.300d')
    #     token_ids, lengths = field.process([examples[0].title, examples[1].title])

    ## Dump the graph and the datasets

    dgl.save_graphs(os.path.join(out_directory, "train_g.bin"), train_g)

    dataset = {
        "val-matrix": val_matrix,
        "test-matrix": test_matrix,
        "item-texts": movie_textual_dataset,
        "item-images": None,
        "user-type": "user",
        "item-type": "movie",
        "user-to-item-type": "watched",
        "item-to-user-type": "watched-by",
        "timestamp-edge-column": "timestamp",
    }

    with open(os.path.join(out_directory, "data.pkl"), "wb") as f:
        pickle.dump(dataset, f)
