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

import math

def get_compatible_batch_size(batch_size, neg_sample_size):
    if neg_sample_size < batch_size and batch_size % neg_sample_size != 0:
        old_batch_size = batch_size
        batch_size = int(math.ceil(batch_size / neg_sample_size) * neg_sample_size)
        print('batch size ({}) is incompatible to the negative sample size ({}). Change the batch size to {}'.format(
            old_batch_size, neg_sample_size, batch_size))
    return batch_size
