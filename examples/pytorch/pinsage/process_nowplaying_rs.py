# Credit: Linfang He (@Linfang-He)
"""
Context attributes:
user_id,
track_id,instrumentalness,liveness,speechiness,danceability,valence,loudness,tempo,acousticness,energy,mode,key,
hashtag,created_at,score,lang,tweet_lang,time_zone,rating

shape:
3,613,460 ratings (listing events)
4,776   users
22,092  items (tracks)

The author used the timestamps to split #nowplaying- RS into the training (from Jan. 1 to Sep. 30) and test sets (from Nov. 1 to Dec. 23). 
To be consistent with movielens, this code combines train and test dataset, then split it into train, validation, and test with the same method
------- not sure if it is necessary

Users who have listened to less than 10 tracks and tracks which have been listened to by less than 10 users are already removed
LEs that do not contain hashtags or do not exhibit any sentiment information from the dataset for the experiments are removed as well

There are only positive examples in original datasets.
To create negative samples, please follow https://github.com/asmitapoddar/nowplaying-RS-Music-Reco-FM/tree/6ab9e65f2c08e6c5733ba40d9f84b3dfa2671fd5 
For each listening event, add nine tracks as the negative samples. 

Use random population (POP_RND), where we added nine randomly chosen tracks that the user has not listened to previously.
"""

import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import dgl
import torch
import torchtext
from builder import PandasGraphBuilder

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str)
parser.add_argument('output_path', type=str)
args = parser.parse_args()
directory = args.directory
output_path = args.output_path

train =  pd.read_csv("Datasets/nowplaying-rs/Context/train_final_POP_RND.txt",sep='\t',header=None,index_col=False,
                  names=['user_id', 'track_id', 'hashtag', 'created_at', 'score', 'lang','tweet_lang', \
                         'time_zone','instrumentalness', 'liveness','speechiness', 'danceability', 'valence', \
                         'loudness', 'tempo','acousticness', 'energy', 'mode', 'key', 'rating'])

test =  pd.read_csv("Datasets/nowplaying-rs/Context/test_final_POP_RND.txt",sep='\t',header=None,index_col=False,
                  names=['user_id', 'track_id', 'hashtag', 'created_at', 'score', 'lang','tweet_lang', \
                         'time_zone','instrumentalness', 'liveness','speechiness', 'danceability', 'valence', \
                         'loudness', 'tempo','acousticness', 'energy', 'mode', 'key', 'rating'])



df = pd.concat([train, test])

users = pd.DataFrame(df['user_id']).astype('category').drop_duplicates()

tracks = train.drop(['user_id', 'hashtag', 'created_at', 'score', 'lang', 'tweet_lang', \
                           'time_zone', 'rating'], axis=1).drop_duplicates()
tracks = pd.DataFrame(tracks).astype({'key': 'category'})

ratings = train.drop(['instrumentalness','liveness','speechiness','danceability', \
                            'valence','loudness','tempo','acousticness','energy','mode','key'],axis=1)
ratings = pd.DataFrame(ratings).astype({'hashtag': 'category', 'lang': 'category', 'tweet_lang': 'category', 'time_zone': 'category'})
ratings['created_at'] = pd.to_datetime(ratings['created_at'],format='%Y-%m-%d %H:%M:%S')

genre_columns = tracks.columns.drop(['track_id', 'key'])


graph_builder = PandasGraphBuilder()
graph_builder.add_entities(users, 'user_id', 'user')
graph_builder.add_entities(tracks, 'track_id', 'track')
graph_builder.add_binary_relations(ratings, 'user_id', 'track_id', 'listened')
graph_builder.add_binary_relations(ratings, 'track_id', 'user_id', 'listened-by')

g = graph_builder.build()


# Assign features.
# Note that variable-sized features such as texts or images are handled elsewhere.
# g.nodes['user'].data['gender'] = torch.LongTensor(users['gender'].cat.codes.values)
# g.nodes['user'].data['age'] = torch.LongTensor(users['age'].cat.codes.values)
# g.nodes['user'].data['occupation'] = torch.LongTensor(users['occupation'].cat.codes.values)
# g.nodes['user'].data['zip'] = torch.LongTensor(users['zip'].cat.codes.values)

g.nodes['track'].data['key'] = torch.LongTensor(tracks['key'].cat.codes.values)
g.nodes['track'].data['genre'] = torch.FloatTensor(tracks[genre_columns].values)

g.edges['listened'].data['hashtag'] = torch.LongTensor(ratings['hashtag'].cat.codes.values)
g.edges['listened'].data['created_at'] = torch.FloatTensor(ratings['created_at'].values.astype('int64') / 1000000000)
g.edges['listened'].data['score'] = torch.FloatTensor(ratings['score'].values)
g.edges['listened'].data['lang'] = torch.LongTensor(ratings['lang'].cat.codes.values)
g.edges['listened'].data['tweet_lang'] = torch.LongTensor(ratings['tweet_lang'].cat.codes.values)
g.edges['listened'].data['time_zone'] = torch.LongTensor(ratings['time_zone'].cat.codes.values)
g.edges['listened'].data['rating'] = torch.LongTensor(ratings['rating'].values)

g.edges['listened-by'].data['hashtag'] = torch.LongTensor(ratings['hashtag'].cat.codes.values)
g.edges['listened-by'].data['created_at'] = torch.FloatTensor(ratings['created_at'].values.astype('int64'))
g.edges['listened-by'].data['score'] = torch.LongTensor(ratings['score'].values)
g.edges['listened-by'].data['lang'] = torch.LongTensor(ratings['lang'].cat.codes.values)
g.edges['listened-by'].data['tweet_lang'] = torch.LongTensor(ratings['tweet_lang'].cat.codes.values)
g.edges['listened-by'].data['time_zone'] = torch.LongTensor(ratings['time_zone'].cat.codes.values)
g.edges['listened-by'].data['rating'] = torch.LongTensor(ratings['rating'].values)


# Train-validation-test split
# This is a little bit tricky as we want to select the last interaction for test, and the
# second-to-last interaction for validation, except that the 
n_edges = g.number_of_edges('listened')
with g.local_scope():
    def splits(edges):
        num_edges, count = edges.data['train_mask'].shape

        # sort by timestamp
        _, sorted_idx = edges.data['created_at'].sort(1)

        train_mask = edges.data['train_mask']
        val_mask = edges.data['val_mask']
        test_mask = edges.data['test_mask']

        x = torch.arange(num_edges)

        # If one user has more than one interactions, select the latest one for test.
        if count > 1:
            train_mask[x, sorted_idx[:, -1]] = False
            test_mask[x, sorted_idx[:, -1]] = True
        # If one user has more than two interactions, select the second latest one for validation.
        if count > 2:
            train_mask[x, sorted_idx[:, -2]] = False
            val_mask[x, sorted_idx[:, -2]] = True
        return {'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask}

    g.edges['listened'].data['train_mask'] = torch.ones(n_edges, dtype=torch.bool)
    g.edges['listened'].data['val_mask'] = torch.zeros(n_edges, dtype=torch.bool)
    g.edges['listened'].data['test_mask'] = torch.zeros(n_edges, dtype=torch.bool)
    g.nodes['track'].data['count'] = g.in_degrees(etype='listened')
    g.group_apply_edges('src', splits, etype='listened')

    train_indices = g.filter_edges(lambda edges: edges.data['train_mask'], etype='listened')
    val_indices = g.filter_edges(lambda edges: edges.data['val_mask'], etype='listened')
    test_indices = g.filter_edges(lambda edges: edges.data['test_mask'], etype='listened')


# Build the graph with training interactions only.
train_g = g.edge_subgraph(
    {'listened': train_indices, 'listened-by': train_indices},
    preserve_nodes=True)
del train_g.nodes['track'].data[dgl.NID]      # remove the induced node IDs - should be assigned by model instead
del train_g.nodes['user'].data[dgl.NID]       # remove the induced user IDs - shoule be assigned by model instead
for ntype in g.ntypes:
    for col, data in g.nodes[ntype].data.items():
        train_g.nodes[ntype].data[col] = data
for etype in g.etypes:
    for col, data in g.edges[etype].data.items():
        train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]


# Build the user-item sparse matrix for validation and test set.
n_users = g.number_of_nodes('user')
n_items = g.number_of_nodes('track')
val_src, val_dst = g.find_edges(val_indices, etype='listened')
test_src, test_dst = g.find_edges(test_indices, etype='listened')
val_src = val_src.numpy()
val_dst = val_dst.numpy()
test_src = test_src.numpy()
test_dst = test_dst.numpy()
val_matrix = ssp.coo_matrix((np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items))
test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))




## Build title set

# movie_textual_dataset = {'title': movies['title'].values}

# The model should build their own vocabulary and process the texts.  Here is one example
# of using torchtext to pad and numericalize a batch of strings.
#     field = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
#     examples = [torchtext.data.Example.fromlist([t], [('title', title_field)]) for t in texts]
#     titleset = torchtext.data.Dataset(examples, [('title', title_field)])
#     field.build_vocab(titleset.title, vectors='fasttext.simple.300d')
#     token_ids, lengths = field.process([examples[0].title, examples[1].title])

## Dump the graph and the datasets

dataset = {
    'train-graph': train_g,
    'val-matrix': val_matrix,
    'test-matrix': test_matrix,
    'item-texts': None,
    'item-images': None,
    'user-type': 'user',
    'item-type': 'track',
    'user-to-item-type': 'listened',
    'item-to-user-type': 'listened-by',
    'timestamp-edge-column': 'created_at'}

with open(output_path, 'wb') as f:
    pickle.dump(dataset, f)
