from functools import partial

class UserProductDataset(object):
    def split_user(self, df, filter_counts=False):
        df_new = df.copy()
        df_new['prob'] = 0

        if filter_counts:
            df_new_sub = (df_new['product_count'] >= 10).nonzero()[0]
        else:
            df_new_sub = df_new['train'].nonzero()[0]
        prob = np.linspace(0, 1, df_new_sub.shape[0], endpoint=False)
        np.random.shuffle(prob)
        df_new['prob'].iloc[df_new_sub] = prob
        return df_new

    def data_split(self):
        self.ratings = self.ratings.groupby('user_id', group_keys=False).apply(
                partial(self.split_user, filter_counts=True))
        self.ratings['train'] = self.ratings['prob'] <= 0.8
        self.ratings['valid'] = (self.ratings['prob'] > 0.8) & (self.ratings['prob'] <= 0.9)
        self.ratings['test'] = self.ratings['prob'] > 0.9
        self.ratings.drop(['prob'], axis=1, inplace=True)

    def find_neighbors(self, restart_prob, max_nodes, top_T):
        # TODO: replace with more efficient PPR estimation
        neighbor_probs, neighbors = randomwalk.random_walk_distribution_topt(
                self.g, self.g.nodes(), restart_prob, max_nodes, top_T)

        self.user_neighbors = []
        for i in range(len(self.user_ids)):
            user_neighbor = neighbors[i]
            self.user_neighbors.append(user_neighbor.tolist())

        self.product_neighbors = []
        for i in range(len(self.user_ids), len(self.user_ids) + len(self.product_ids)):
            product_neighbor = neighbors[i]
            self.product_neighbors.append(product_neighbor.tolist())

    def generate_mask(self):
        while True:
            ratings = self.ratings.groupby('user_id', group_keys=False).apply(self.split_user)
            prior_prob = ratings['prob'].values
            for i in range(5):
                train_mask = (prior_prob >= 0.2 * i) & (prior_prob < 0.2 * (i + 1))
                prior_mask = ~train_mask
                train_mask &= ratings['train'].values
                prior_mask &= ratings['train'].values
                yield prior_mask, train_mask

    def refresh_mask(self):
        if not hasattr(self, 'masks'):
            self.masks = self.generate_mask()
        prior_mask, train_mask = next(self.masks)

        valid_tensor = torch.from_numpy(self.ratings['valid'].values.astype('uint8'))
        test_tensor = torch.from_numpy(self.ratings['test'].values.astype('uint8'))
        train_tensor = torch.from_numpy(train_mask.astype('uint8'))
        prior_tensor = torch.from_numpy(prior_mask.astype('uint8'))
        edge_data = {
                'prior': prior_tensor,
                'valid': valid_tensor,
                'test': test_tensor,
                'train': train_tensor,
                }

        self.g.edges[self.rating_user_vertices, self.rating_product_vertices].data.update(edge_data)
        self.g.edges[self.rating_product_vertices, self.rating_user_vertices].data.update(edge_data)
