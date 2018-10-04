def cycle(loader):
    while True:
        for x in loader:
            yield x
