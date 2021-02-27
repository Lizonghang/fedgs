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

    def train(self, my_round):
        """Trains on self.model using one batch of train_data.
        Args:
            my_round: The current training round, used for learning rate
                decay.
        Returns:
            comp: Number of FLOPs executed in training process.
            num_samples: Number of train samples on this client.
            update: Trained model params.
        """
        comp, update = self.model.train(self.train_data_iter, my_round)
        return comp, self.num_train_samples, update

    def test(self, set_to_use="test"):
        """Tests self.model on self.test_data.
        Args:
            set_to_use: Set to test on. Should be in ["train", "test"].
        Returns:
            metrics: Dict of metrics returned by the model.
        """
        assert set_to_use in ["train", "test", "val"]
        if set_to_use == "train":
            data = self.train_data
        elif set_to_use == "test" or set_to_use == "val":
            data = self.test_data
        return self.model.test(data)

    def set_model(self, model):
        """Set the model data to specified model.
        Args:
            model: The specified model.
        """
        self.model.set_params(model.get_params())

    @property
    def num_train_samples(self):
        """Return the number of train samples for this client."""
        return len(self.train_data["y"])

    @property
    def num_test_samples(self):
        """Return the number of test samples for this client."""
        return len(self.test_data["y"])

    @property
    def num_samples(self):
        """Return the number of train + test samples for this client."""
        return self.num_train_samples + self.num_test_samples

    @property
    def train_sample_dist(self):
        """Return the distribution of train data for this client."""
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
        """Return the distribution of test data for this client."""
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
        """Return the distribution of overall data for this client."""
        return self.train_sample_dist + self.test_sample_dist

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn("The current implementation shares the model among all clients."
                      "Setting it on one client will effectively modify all clients.")
        self._model = model

    def process_data(self, data):
        """Convert train data and test data to NDArray objects with
        specified context.
        Args:
            data: List of train vectors or labels.
        Returns:
            nd_data: Format NDArray data with specified context.
        """
        return nd.array(data, ctx=self.model.ctx)
