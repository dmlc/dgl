# -*- coding: utf-8 -*-
#
# setup.py
#
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from models import KEModel

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd

import os
import logging
import time
import json

def load_model(logger, args, n_entities, n_relations, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel)
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"

    logger.info('Load model {}'.format(args.model_name))
    return model

def load_model_from_checkpoint(logger, args, n_entities, n_relations, ckpt_path):
    model = load_model(logger, args, n_entities, n_relations)
    model.load_emb(ckpt_path, args.dataset)
    return model

def train(args, model, train_sampler, valid_samplers=None, rank=0, rel_parts=None, barrier=None):
    assert args.num_proc <= 1, "MXNet KGE does not support multi-process now"
    assert args.rel_part == False, "No need for relation partition in single process for MXNet KGE"
    logs = []

    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part:
        model.prepare_relation(mx.gpu(gpu_id))

    start = time.time()
    for step in range(0, args.max_step):
        pos_g, neg_g = next(train_sampler)
        args.step = step
        with mx.autograd.record():
            loss, log = model.forward(pos_g, neg_g, gpu_id)
        loss.backward()
        logs.append(log)
        model.update(gpu_id)

        if step % args.log_interval == 0:
            for k in logs[0].keys():
                v = sum(l[k] for l in logs) / len(logs)
                print('[Train]({}/{}) average {}: {}'.format(step, args.max_step, k, v))
            logs = []
            print(time.time() - start)
            start = time.time()

        if args.valid and step % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            start = time.time()
            test(args, model, valid_samplers, mode='Valid')
            print('test:', time.time() - start)
    if args.strict_rel_part:
        model.writeback_relation(rank, rel_parts)

    # clear cache
    logs = []

def test(args, model, test_samplers, rank=0, mode='Test', queue=None):
    assert args.num_proc <= 1, "MXNet KGE does not support multi-process now"
    logs = []

    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part:
        model.load_relation(mx.gpu(gpu_id))

    for sampler in test_samplers:
        #print('Number of tests: ' + len(sampler))
        count = 0
        for pos_g, neg_g in sampler:
            model.forward_test(pos_g, neg_g, logs, gpu_id)

    metrics = {}
    if len(logs) > 0:
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

    for k, v in metrics.items():
        print('{} average {}: {}'.format(mode, k, v))
    for i in range(len(test_samplers)):
        test_samplers[i] = test_samplers[i].reset()
