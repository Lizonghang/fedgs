from abc import ABC, abstractmethod
import numpy as np
import mxnet as mx
import warnings

from mxnet import autograd, nd
from baseline_constants import INPUT_SIZE


class Model(ABC):

    def __init__(self, seed, lr, ctx, optimizer=None, count_ops=False):
        self.lr = lr
        self.seed = seed
        self._optimizer = optimizer
        self.ctx = ctx
        self.count_ops = count_ops

        mx.random.seed(123 + self.seed)
        np.random.seed(self.seed)

        self.net, self.loss, self.trainer = self.create_model()

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = "sgd"

        return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.
        Returns:
            A 3-tuple consisting of:
                net: A neural network workflow.
                loss: An operation that, when run with the features and the
                    labels, computes the loss value.
                train_op: An operation that, when grads are computed, trains
                    the model.
        """
        return None, None, None

    def train(self, data_iter):
        """
        Trains the client model.
        Args:
            data: Dict of the form {'x': NDArray, 'y': NDArray}.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
        """
        batched_x, batched_y = next(data_iter)
        input_data = self.preprocess_x(batched_x)
        target_data = self.preprocess_y(batched_y)
        num_samples = len(batched_y)

        with autograd.record():
            y_hats = self.net(input_data)
            ls = self.loss(y_hats, target_data)
            ls.backward()
        self.trainer.step(num_samples)
        nd.waitall()

        update = self.get_params()
        comp = num_samples * self.flops if self.count_ops else 0
        return comp, num_samples, update

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
        params = self.net.collect_params().values()
        tot_size = 0
        for p in params:
            tot_elems = self.__num_elems(p.shape)
            dtype_size = np.dtype(p.dtype).itemsize
            var_size = tot_elems * dtype_size
            tot_size += var_size
        return tot_size

    @property
    def flops(self):
        """Returns the number of flops needed to propagate a sample through the
        network.
        The package MXOP is required:
            https://github.com/hey-yahei/OpSummary.MXNet
        Run "pip install --index-url https://pypi.org/simple/ mxop" to install.
        MXOP can only run on cpu device, so the context is reset to cpu and then
        reset back to the specified device.
        If MXOP is not installed, 0 will be returned.
        Return:
            integer representing the number of flops
        """
        try:
            from mxop.gluon import count_ops
            self.set_context(mx.cpu())
            op_counter = count_ops(self.net, (1, *INPUT_SIZE))
            self.set_context(self.ctx)
            return sum(op_counter.values())
        except ModuleNotFoundError:
            warnings.warn("MXOP is not installed, num_flops=0 is returned.")
            return 0

    def set_params(self, model_params):
        source_params = list(model_params)
        target_params = list(self.get_params())
        num_params = len(target_params)
        for p in range(num_params):
            target_params[p].set_data(source_params[p].data())

    def get_params(self):
        return self.net.collect_params().values()

    def set_context(self, ctx):
        self.net.collect_params().reset_ctx(ctx)

    @abstractmethod
    def test(self, data):
        """
        Tests the current model on the given data.
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        return None

    @abstractmethod
    def preprocess_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return None

    @abstractmethod
    def preprocess_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return None
