from functools import partial
import numpy as np
from .. import randomwalk
import tqdm

class UserProductDataset(object):
    def split_user(self, df, filter_counts=0, timestamp=None):
        df_new = df.copy()
        df_new['prob'] = -1

        df_new_sub = (df_new['product_count'] >= filter_counts).to_numpy().nonzero()[0]
        prob = np.linspace(0, 1, df_new_sub.shape[0], endpoint=False)
        if timestamp is not None and timestamp in df_new.columns:
            df_new = df_new.iloc[df_new_sub].sort_values(timestamp)
            df_new['prob'] = prob
            return df_new
        else:
            np.random.shuffle(prob)
            df_new['prob'].iloc[df_new_sub] = prob
            return df_new

    def data_split(self, ratings):
        ratings = ratings.groupby('user_id', group_keys=False).apply(
                partial(self.split_user, filter_counts=10, timestamp='timestamp'))
        ratings['train'] = ratings['prob'] <= 0.8
        ratings['valid'] = (ratings['prob'] > 0.8) & (ratings['prob'] <= 0.9)
        ratings['test'] = ratings['prob'] > 0.9
        ratings.drop(['prob'], axis=1, inplace=True)
        return ratings

    def find_neighbors(self, restart_prob, max_nodes, top_T, batch_size=1024):
        # TODO: replace with more efficient PPR estimation
        import torch
        self.user_neighbors = []
        self.product_neighbors = []

        for i in tqdm.trange(0, len(self.user_ids), batch_size):
            batch = torch.arange(i, min(i + batch_size, len(self.user_ids)))
            neighbor_probs, neighbors = randomwalk.random_walk_distribution_topt(
                    self.g, batch, restart_prob, max_nodes, top_T)
            self.user_neighbors.extend(list(neighbors))

        for i in tqdm.trange(0, len(self.product_ids), batch_size):
            batch = torch.arange(i, min(i + batch_size, len(self.product_ids)))
            neighbor_probs, neighbors = randomwalk.random_walk_distribution_topt(
                    self.g, batch, restart_prob, max_nodes, top_T)
            self.product_neighbors.extend(list(neighbors))

    def generate_mask(self):
        while True:
            ratings = self.ratings.groupby('user_id', group_keys=False).apply(
                    partial(self.split_user, filter_counts=5, timestamp='timestamp'))
            prior_prob = ratings['prob'].values
            for i in range(5):
                train_mask = (prior_prob >= 0.2 * i) & (prior_prob < 0.2 * (i + 1))
                prior_mask = ~train_mask
                train_mask &= ratings['train'].values
                prior_mask &= ratings['train'].values
                yield prior_mask, train_mask

    def refresh_mask(self):
        import torch
        if not hasattr(self, 'masks'):
            self.masks = self.generate_mask()
        prior_mask, train_mask = next(self.masks)

        valid_tensor = torch.from_numpy(self.ratings['valid'].values.astype('uint8'))
        test_tensor = torch.from_numpy(self.ratings['test'].values.astype('uint8'))
        train_tensor = torch.from_numpy(train_mask.astype('uint8'))
        prior_tensor = torch.from_numpy(prior_mask.astype('uint8'))
        edge_data = {
                'prior': torch.cat([prior_tensor, prior_tensor], 0),
                'valid': torch.cat([valid_tensor, valid_tensor], 0),
                'test': torch.cat([test_tensor, test_tensor], 0),
                'train': torch.cat([train_tensor, train_tensor], 0),
                }

        # not supporting slice
        self.g.edges[torch.arange(0, len(self.ratings) * 2)].data.update(edge_data)
        #self.g.edges[torch.arange(len(self.ratings) * 2, self.g.number_of_edges())].data['prior'] = \
        #        torch.ones(self.g.number_of_edges() - len(self.ratings) * 2).byte()

    # Generate the list of products for each user in training/validation/test set.
    def generate_candidates(self):
        self.p_train = []
        self.p_valid = []
        self.p_test = []
        for uid in tqdm.tqdm(self.user_ids):
            user_ratings = self.ratings[self.ratings['user_id'] == uid]
            self.p_train.append(np.array(
                [self.product_ids_invmap[i] for i in user_ratings[user_ratings['train']]['product_id'].values]))
            self.p_valid.append(np.array(
                [self.product_ids_invmap[i] for i in user_ratings[user_ratings['valid']]['product_id'].values]))
            self.p_test.append(np.array(
                [self.product_ids_invmap[i] for i in user_ratings[user_ratings['test']]['product_id'].values]))
