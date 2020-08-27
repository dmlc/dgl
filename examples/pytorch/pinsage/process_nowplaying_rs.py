"""
Script that reads from raw Nowplaying-RS data and dumps into a pickle
file a heterogeneous graph with categorical and numeric features.
"""

import os
import argparse
import pandas as pd
import scipy.sparse as ssp
import pickle
from data_utils import *
from builder import PandasGraphBuilder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    directory = args.directory
    output_path = args.output_path

    data = pd.read_csv(os.path.join(directory, 'context_content_features.csv'))
    track_feature_cols = list(data.columns[1:13])
    data = data[['user_id', 'track_id', 'created_at'] + track_feature_cols].dropna()

    users = data[['user_id']].drop_duplicates()
    tracks = data[['track_id'] + track_feature_cols].drop_duplicates()
    assert tracks['track_id'].value_counts().max() == 1
    tracks = tracks.astype({'mode': 'int64', 'key': 'int64', 'artist_id': 'category'})
    events = data[['user_id', 'track_id', 'created_at']]
    events['created_at'] = events['created_at'].values.astype('datetime64[s]').astype('int64')

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'user_id', 'user')
    graph_builder.add_entities(tracks, 'track_id', 'track')
    graph_builder.add_binary_relations(events, 'user_id', 'track_id', 'listened')
    graph_builder.add_binary_relations(events, 'track_id', 'user_id', 'listened-by')

    g = graph_builder.build()

    float_cols = []
    for col in tracks.columns:
        if col == 'track_id':
            continue
        elif col == 'artist_id':
            g.nodes['track'].data[col] = torch.LongTensor(tracks[col].cat.codes.values)
        elif tracks.dtypes[col] == 'float64':
            float_cols.append(col)
        else:
            g.nodes['track'].data[col] = torch.LongTensor(tracks[col].values)
    g.nodes['track'].data['song_features'] = torch.FloatTensor(linear_normalize(tracks[float_cols].values))
    g.edges['listened'].data['created_at'] = torch.LongTensor(events['created_at'].values)
    g.edges['listened-by'].data['created_at'] = torch.LongTensor(events['created_at'].values)

    n_edges = g.number_of_edges('listened')
    train_indices, val_indices, test_indices = train_test_split_by_time(events, 'created_at', 'user_id')
    train_g = build_train_graph(g, train_indices, 'user', 'track', 'listened', 'listened-by')
    assert train_g.out_degrees(etype='listened').min() > 0
    val_matrix, test_matrix = build_val_test_matrix(
        g, val_indices, test_indices, 'user', 'track', 'listened')

    dataset = {
        'train-graph': train_g,
        'val-matrix': val_matrix,
        'test-matrix': test_matrix,
        'item-texts': {},
        'item-images': None,
        'user-type': 'user',
        'item-type': 'track',
        'user-to-item-type': 'listened',
        'item-to-user-type': 'listened-by',
        'timestamp-edge-column': 'created_at'}

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
