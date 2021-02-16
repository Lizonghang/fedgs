import os
import numpy as np
from mxnet import init

from utils.model_utils import build_net


class ServerModel:

    def __init__(self, client_model, dataset, model_name, num_classes, ctx):
        self.client_model = client_model
        self.dataset = dataset
        self.model_name = model_name
        self.num_classes = num_classes
        self.ctx = ctx
        self.create_model()

    def create_model(self):
        """build and synchronize the server model"""
        self.model = build_net(
            self.dataset, self.model_name, self.num_classes, self.ctx)

        if self.client_model:
            self.set_params(self.client_model.get_params())
        else:
            self.model.initialize(
                init.Zero(), ctx=self.ctx, force_reinit=True)

    def set_params(self, model_params):
        source_params = list(model_params)
        target_params = list(self.get_params())
        num_params = len(target_params)
        for p in range(num_params):
            target_params[p].set_data(source_params[p].data())

    def get_params(self):
        return self.model.collect_params().values()

    def __num_elems(self, shape):
        '''Returns the number of elements in the given shape
        Args:
            shape: Parameter shape
        Return:
            tot_elems: int
        '''
        tot_elems = 1
        for s in shape:
            tot_elems *= int(s)
        return tot_elems

    @property
    def size(self):
        '''Returns the size of the network in bytes
        The size of the network is calculated by summing up the sizes of each
        trainable variable. The sizes of variables are calculated by multiplying
        the number of bytes in their dtype with their number of elements, captured
        in their shape attribute
        Return:
            integer representing size of graph (in bytes)
        '''
        params = self.model.collect_params().values()
        tot_size = 0
        for p in params:
            tot_elems = self.__num_elems(p.shape)
            dtype_size = np.dtype(p.dtype).itemsize
            var_size = tot_elems * dtype_size
            tot_size += var_size
        return tot_size

    def save_model(self, log_dir):
        """Saves the server model to:
            {log_dir}/{self.model_name}.params
        """
        self.model.save_parameters(
            os.path.join(log_dir, self.model_name + ".params"))
