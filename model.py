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
        """Optimizer to be used, the default is SGD optimizer."""
        if self._optimizer is None:
            self._optimizer = "sgd"

        return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.
        Returns:
            A 3-tuple consisting of:
                net: A neural network.
                loss: An operation to compute the loss value.
                train_op: An operation to train the model.
        """
        return None, None, None

    def train(self, data_iter):
        """
        Train the model using a batch of data.
        Args:
            data_iter: An iterator to generate batches of train data.
        Returns:
            comp: Number of FLOPs computed while training given data.
                If --count-op is not set, FLOPs = 0 will be returned.
            num_samples: Number of samples used to train given data.
            update: The model after training given data.
        """
        batched_x, batched_y = next(data_iter)
        input_data = self.preprocess_x(batched_x)
        target_data = self.preprocess_y(batched_y)
        num_samples = len(batched_y)

        # Set MXNET_ENFORCE_DETERMINISM=1 to avoid difference in
        # calculation precision.
        with autograd.record():
            y_hats = self.net(input_data)
            ls = self.loss(y_hats, target_data)
            ls.backward()
        self.trainer.step(num_samples)
        # Wait to avoid running out of GPU memory
        nd.waitall()

        update = self.get_params()
        comp = num_samples * self.flops if self.count_ops else 0
        return comp, num_samples, update

    def __num_elems(self, shape):
        """Returns the number of elements in the given shape.
        Args:
            shape: Parameter shape.
        Return:
            tot_elems: Number of elements.
        """
        tot_elems = 1
        for s in shape:
            tot_elems *= int(s)
        return tot_elems

    @property
    def size(self):
        """Returns the size of the network in bytes.
        The size of the network is calculated by summing up the sizes of each
        trainable variable. The sizes of variables are calculated by multiplying
        the number of bytes in their dtype with their number of elements, captured
        in their shape attribute.
        Return:
            tot_size: Integer representing size of neural network (in bytes).
        """
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
        Note that "pip install --index-url https://pypi.org/simple/ mxop" may
        change the version of the dependent package. Due to MXOP runs on CPU,
        the context is reset to cpu and then reset back to the specified device.
        If MXOP is not installed, 0 will be directly returned.
        Return:
            flops: Integer representing the number of flops.
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
        """Set current model data to given model data.
        Args:
            model_params: Given model data.
        """
        source_params = list(model_params)
        target_params = list(self.get_params())
        num_params = len(target_params)
        for p in range(num_params):
            target_params[p].set_data(source_params[p].data())

    def get_params(self):
        """Return current model data.
        Returns:
            params: Current model data.
        """
        return self.net.collect_params().values()

    def set_context(self, ctx):
        """Move current model to the specified context.
        Args:
            ctx: The specified CPU or GPU context.
        """
        self.net.collect_params().reset_ctx(ctx)

    @abstractmethod
    def test(self, data):
        """Tests the current model on the given data.
        Args:
            data: Dict of the form {"x": NDArray, "y": NDArray}
        Returns:
            stat_metrics: dict of metrics that will be recorded
                by the simulation.
        """
        return None

    @abstractmethod
    def preprocess_x(self, raw_x_batch):
        """Pre-processes each batch of train data before being
            fed to the model."""
        return None

    @abstractmethod
    def preprocess_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed
            to the model."""
        return None
