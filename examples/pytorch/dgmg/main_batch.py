"""
Learning Deep Generative Models of Graphs
Paper: https://arxiv.org/pdf/1803.03324.pdf
This implementation works with a minibatch of size larger than 1 for training and 1 for inference.
"""
import argparse
import datetime
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from model_batch import DGMG


def main(opts):
    t1 = time.time()

    # Setup dataset and data loader
    if opts['dataset'] == 'cycles':
        from cycles import CycleDataset, CycleModelEvaluation, CyclePrinting

        dataset = CycleDataset(fname=opts['path_to_dataset'])
        evaluator = CycleModelEvaluation(v_min=opts['min_size'],
                                         v_max=opts['max_size'],
                                         dir = opts['log_dir'])
        printer = CyclePrinting(num_epochs=opts['nepochs'],
                                num_batches=len(dataset) // opts['batch_size'])
    else:
        raise ValueError('Unsupported dataset: {}'.format(opts['dataset']))

    data_loader = DataLoader(dataset, batch_size=opts['batch_size'], shuffle=True, num_workers=0,
                             collate_fn=dataset.collate_batch)

    # Initialize_model
    model = DGMG(v_max=opts['max_size'],
                 node_hidden_size=opts['node_hidden_size'],
                 num_prop_rounds=opts['num_propagation_rounds'])

    # Initialize optimizer
    if opts['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=opts['lr'])
    else:
        raise ValueError('Unsupported argument for the optimizer')

    t2 = time.time()

    # Training
    model.train()
    for epoch in range(opts['nepochs']):
        for batch, data in enumerate(data_loader):

            log_prob = model(batch_size=opts['batch_size'], actions=data)

            loss = - log_prob / opts['batch_size']
            batch_avg_prob = (log_prob / opts['batch_size']).detach().exp()
            batch_avg_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            if opts['clip_grad']:
                clip_grad_norm_(model.parameters(), opts['clip_bound'])
            optimizer.step()

            printer.update(epoch + 1,  {'averaged loss': batch_avg_loss,
                                        'averaged prob': batch_avg_prob})

    t3 = time.time()

    model.eval()
    evaluator.rollout_and_examine(model, opts['num_generated_samples'])
    evaluator.write_summary()

    t4 = time.time()

    print('It took {} to setup.'.format(datetime.timedelta(seconds=t2-t1)))
    print('It took {} to finish training.'.format(datetime.timedelta(seconds=t3-t2)))
    print('It took {} to finish evaluation.'.format(datetime.timedelta(seconds=t4-t3)))
    print('--------------------------------------------------------------------------')
    print('On average, an epoch takes {}.'.format(datetime.timedelta(
        seconds=(t3-t2) / opts['nepochs'])))

    del model.g_list
    torch.save(model, './model_batched.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='batched DGMG')

    # configure
    parser.add_argument('--seed', type=int, default=9284, help='random seed')

    # dataset
    parser.add_argument('--dataset', choices=['cycles'], default='cycles',
                        help='dataset to use')
    parser.add_argument('--path-to-dataset', type=str, default='cycles.p',
                        help='load the dataset if it exists, '
                             'generate it and save to the path otherwise')

    # log
    parser.add_argument('--log-dir', default='./results',
                        help='folder to save info like experiment configuration '
                             'or model evaluation results')

    # optimization
    parser.add_argument('--batch-size', type=int, default=10,
                        help='batch size to use for training')
    parser.add_argument('--clip-grad', action='store_true', default=True,
                        help='gradient clipping is required to prevent gradient explosion')
    parser.add_argument('--clip-bound', type=float, default=0.25,
                        help='constraint of gradient norm for gradient clipping')

    args = parser.parse_args()
    from utils import setup
    opts = setup(args)

    main(opts)