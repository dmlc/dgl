import dgl.data as data

def test_minigc():
    ds = data.MiniGCDataset(16, 10, 20)
    g, l = list(zip(*ds))
    print(g, l)

if __name__ == '__main__':
    test_minigc()
