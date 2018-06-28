import torch

def pad_and_mask(orderings):
    batch_size = len(orderings)
    heads = [0] * batch_size
    lengths = map(len, orderings)

    actions = []
    select_nodes = []
    masks = []

    phase = 0 # add node
    while True:
        mask = []
        selected = []
        if phase == 0: # add node phase

            # check if more actions available
            finished = True
            for head, length, ordering in zip(heads, lengths, orderings):
                if head < length:
                    assert(isinstance(ordering[head], int))
                    finished = False
            if finished: # every head == length
                break

            actions.append(0) # add node

            for batch_idx, (head, length, ordering) in enumerate(zip(heads, lengths, orderings)):
                if head < length:
                    mask.append(1)
                    selected.append(ordering[head]) # record added node
                    head += 1
                    if head < length and isinstance(ordering[head], tuple):
                        phase = 1
                    heads[batch_idx] = head
                else:
                    mask.append(0)
                    selected.append(-1) # pad

        else: # add edge phase
            actions.append(1) # add edge
            phase = 0 # reset to add node phase
            for batch_idx, (head, length, ordering) in enumerate(zip(heads, lengths, orderings)):
                if head < length and isinstance(ordering[head], tuple):
                    mask.append(1)
                    selected.append(ordering[head][0]) # assume edge (u, v), u < v
                    head += 1
                    if head < length and isinstance(ordering[head], tuple):
                        phase = 1 # still add edge phase for next iteration
                    heads[batch_idx] = head
                else:
                    mask.append(0)
                    selected.append(-1) # pad

        assert(len(selected) == batch_size)
        select_nodes.append(selected)
        assert(len(mask) == batch_size)
        masks.append(mask)

    return actions, masks, select_nodes

if __name__ == '__main__':
    # test pad_and_mask

    # graph generation action list
    orderings = [
                 [0, 1, 5, (0, 5), (1, 5), 3, (1, 3)],
                 [7, 1, (7, 1), 3, 6, (7, 6), (3, 6), (1, 6)]
                ]
    for i in orderings:
        print i
    print
    for i in pad_and_mask(orderings):
        print(i)
