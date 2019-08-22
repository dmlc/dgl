from modelnet import ModelNet

dataset = ModelNet('modelnet40-sampled-2048.h5', 32)

class Model(nn.Module):
    def __init__(self):
        super().__init__(self)

    def forward(self, x):
        hs = []
        batch_size, n_points, x_dims = x.shape
        h = x.view(-1, x_dims)
        for i in range(self.num_layers)):
            g = self.nng(h, [n_points] * batch_size)
            h = self.conv[0](g)
            hs.append(h.view(batch_size, n_points, -1))

        h = torch.cat(hs, 2)
        h = self.proj(h)
        h_max = torch.max(h, 1)
        h_avg = torch.mean(h, 1)
        h = torch.cat([h_max, h_avg], 1)

        # TODO: FC
