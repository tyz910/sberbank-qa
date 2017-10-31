# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from .utils import AverageMeter
from .rnn_reader import RnnDocReader


class DocReaderModel(object):
    def __init__(self, opt, embedding):
        self.opt = opt
        self.network = RnnDocReader(opt, embedding=embedding)
        self.optimizer = optim.Adamax([p for p in self.network.parameters() if p.requires_grad], weight_decay=opt['weight_decay'])
        self.train_loss = AverageMeter()

    def load_weights(self, weights):
        state_dict = self.network.state_dict()

        for k, v in weights.items():
            state_dict[k] = v

        self.network.load_state_dict(state_dict)

    def save_weights(self, filename):
        torch.save(self.network.state_dict(), filename)

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:7]]
            target_s = Variable(ex[7].cuda(async=True))
            target_e = Variable(ex[8].cuda(async=True))
        else:
            inputs = [Variable(e) for e in ex[:7]]
            target_s = Variable(ex[7])
            target_e = Variable(ex[8])

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        self.train_loss.update(loss.data[0], ex[0].size(0))

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:7]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:7]]

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Transfer to CPU/normal tensors for numpy ops
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        return score_s, score_e

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def cuda(self):
        self.network.cuda()
