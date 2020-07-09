import dgl.data as data

def test_minigc():
    ds = data.MiniGCDataset(16, 10, 20, save_graph=False, force_reload=True)
    g, l = list(zip(*ds))
    print(g, l)

if __name__ == '__main__':
    test_minigc()
