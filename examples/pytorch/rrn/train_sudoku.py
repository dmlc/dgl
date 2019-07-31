from sudoku_data import sudoku_dataloader
import argparse
from sudoku import SudokuNN
import torch
from torch.optim import Adam
import os
import numpy as np
import logging


def accuracy(preds, target):
    batch_size = preds.size(0)
    right_count = 0
    for i in range(batch_size):
        if torch.all(torch.eq(preds[i], target[i])).__bool__():
            right_count += 1
    return right_count / batch_size


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    if args.do_train:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        logging.basicConfig(filename=os.path.join(args.output_dir, 'train.log'), filemode='w', level=logging.DEBUG)
        # print = logging.info

        model = SudokuNN(num_steps=args.steps, edge_drop=args.edge_drop).to(device)
        train_dataloader = sudoku_dataloader(args.batch_size, segment='train')
        dev_dataloader = sudoku_dataloader(args.batch_size, segment='valid')

        t_total = len(train_dataloader) * args.epochs

        opt = Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)

        best_dev_acc = 0.0
        for epoch in range(args.epochs):
            model.train()
            for i, g in enumerate(train_dataloader):
                g.ndata['q'] = g.ndata['q'].to(device)
                g.ndata['a'] = g.ndata['a'].to(device)
                g.ndata['row'] = g.ndata['row'].to(device)
                g.ndata['col'] = g.ndata['col'].to(device)

                _, loss = model(g)

                opt.zero_grad()
                loss.backward()
                opt.step()
                if i % 100 == 0:
                    print(f"Epoch {epoch}, batch {i}, loss {loss.cpu().data}")

            # dev
            print("\n=========Dev step========")
            model.eval()
            dev_loss = []
            dev_res = []
            for g in dev_dataloader:
                g.ndata['q'] = g.ndata['q'].to(device)
                g.ndata['a'] = g.ndata['a'].to(device)
                g.ndata['row'] = g.ndata['row'].to(device)
                g.ndata['col'] = g.ndata['col'].to(device)

                target = g.ndata['a']
                target = target.view([-1, 81])

                with torch.no_grad():
                    preds, loss = model(g, is_training=False)
                    preds = preds.view([-1, 81])

                    for i in range(preds.size(0)):
                        dev_res.append(int(torch.equal(preds[i, :], target[i, :])))

                    dev_loss.append(loss.cpu().detach().data)

            dev_acc = sum(dev_res) / len(dev_res)
            print(f"Dev loss {np.mean(dev_loss)}, accuracy {dev_acc}")
            if dev_acc >= best_dev_acc:
                torch.save(model, os.path.join(args.output_dir, 'model_best.bin'))
                best_dev_acc = dev_acc
            print(f"Best dev accuracy {best_dev_acc}\n")

        torch.save(model, os.path.join(args.output_dir, 'model_final.bin'))

    if args.do_eval:
        model_path = os.path.join(args.output_dir, 'model_best.bin')
        if not os.path.exists(model_path):
            raise FileNotFoundError("Saved model not Found!")

        model = torch.load(model_path).to(device)
        test_dataloader = sudoku_dataloader(args.batch_size, segment='test')

        print("\n=========Test step========")
        model.eval()
        test_loss = []
        test_res = []
        for g in test_dataloader:
            g.ndata['q'] = g.ndata['q'].to(device)
            g.ndata['a'] = g.ndata['a'].to(device)
            g.ndata['row'] = g.ndata['row'].to(device)
            g.ndata['col'] = g.ndata['col'].to(device)

            target = g.ndata['a']
            target = target.view([-1, 81])

            with torch.no_grad():
                preds, loss = model(g, is_training=False)
                preds = preds
                preds = preds.view([-1, 81])

                for i in range(preds.size(0)):
                    test_res.append(int(torch.equal(preds[i, :], target[i, :])))

                test_loss.append(loss.cpu().detach().data)

        test_acc = sum(test_res) / len(test_res)
        print(f"Test loss {np.mean(test_loss)}, accuracy {test_acc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sudoku')
    parser.add_argument("--output_dir", type=str, default=None, required=True,
                        help="The directory to save model")
    parser.add_argument("--do_train", default=False, action="store_true",
                        help="Train the model")
    parser.add_argument("--do_eval", default=False, action="store_true",
                        help="Evaluate the model on test data")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--edge_drop", type=float, default=0.4,
                        help="Dropout rate at edges.")
    parser.add_argument("--steps", type=int, default=32,
                        help="Number of message passing steps.")

    args = parser.parse_args()

    main(args)


