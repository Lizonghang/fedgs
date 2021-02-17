import warnings
import numpy as np
from mxnet import nd

from utils.model_utils import batch_data


class Client:

    def __init__(self, seed, client_id, group, train_data, test_data, model, batch_size):
        self.seed = seed
        self.id = client_id
        self.group = group
        self._model = model
        self.train_data = {
            "x": self.process_data(train_data["x"]),
            "y": self.process_data(train_data["y"])
        }
        self.test_data = {
            "x": self.process_data(test_data["x"]),
            "y": self.process_data(test_data["y"])
        }
        self.train_data_iter = batch_data(
            self.train_data, batch_size, seed=self.seed)

    def train(self):
        """Trains on self.model using one batch of train_data.
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update:
        """
        comp, num_samples, update = self.model.train(self.train_data_iter)
        return comp, num_samples, update

    def test(self, set_to_use="test"):
        """Tests self.model on self.test_data.
        Args:
            set_to_use. Set to test on. Should be in ["train", "test"].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ["train", "test", "val"]
        if set_to_use == "train":
            data = self.train_data
        elif set_to_use == "test" or set_to_use == "val":
            data = self.test_data
        return self.model.test(data)

    def set_model(self, model):
        self.model.set_params(model.get_params())

    @property
    def num_train_samples(self):
        """Number of train samples for this client.
        Return:
            int: Number of train samples for this client
        """
        return len(self.train_data["y"])

    @property
    def num_test_samples(self):
        """Number of test samples for this client.
        Return:
            int: Number of test samples for this client
        """
        return len(self.test_data["y"])

    @property
    def num_samples(self):
        """Number of samples for this client (train + test).
        Return:
            int: Number of samples for this client
        """
        return self.num_train_samples + self.num_test_samples

    @property
    def train_sample_dist(self):
        labels = self.train_data["y"]
        labels = labels.asnumpy().astype("int64")
        dist = np.bincount(labels)
        # align to num_classes
        num_classes = self.model.num_classes
        dist = np.concatenate(
            (dist, np.zeros(num_classes-len(dist))))
        return dist

    @property
    def test_sample_dist(self):
        labels = self.test_data["y"]
        labels = labels.asnumpy().astype("int64")
        dist = np.bincount(labels)
        # align to num_classes
        num_classes = self.model.num_classes
        dist = np.concatenate(
            (dist, np.zeros(num_classes-len(dist))))
        return dist

    @property
    def sample_dist(self):
        return self.train_sample_dist + self.test_sample_dist

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model

    def process_data(self, data):
        return nd.array(data, ctx=self.model.ctx)
