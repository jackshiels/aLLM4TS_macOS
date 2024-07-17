import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        if self.args.use_multi_gpu and self.args.use_gpu:
            self.device = torch.device("mps")
            print(f"*********************{self.device}***************************")
            self.model = torch.nn.parallel.DistributedDataParallel( self._build_model().to(self.device), device_ids=[self.args.local_rank],output_device=self.args.local_rank)
            
        else:
            self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device("mps")
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
