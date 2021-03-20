import warnings
import numpy as np
from mxnet import nd

from utils.model_utils import batch_data


class Client:

    def __init__(self, seed, client_id, group, train_data,
                 test_data, model, batch_size):
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
        self.batch_iter = batch_data(
            self.train_data, batch_size, seed=self.seed)
        self.next_batch_x, self.next_batch_y = next(self.batch_iter)

    def train(self, my_round):
        """Trains on self.model using one batch of train_data.
        Args:
            my_round: The current training round, used for learning rate
                decay.
        Returns:
            comp: Number of FLOPs executed in training process.
            num_samples: Number of trained batch samples on this client.
            update: Trained model params.
        """
        comp, num_samples, update = self.model.train(
            self.next_batch_x, self.next_batch_y, my_round)

        # Prepare data for next synchronization
        self.next_batch_x, self.next_batch_y = next(self.batch_iter)

        return comp, num_samples, update

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
        if not hasattr(self, "_num_train_samples"):
            self._num_train_samples = len(self.train_data["y"])

        return self._num_train_samples

    @property
    def num_test_samples(self):
        """Return the number of test samples for this client."""
        if not hasattr(self, "_num_test_samples"):
            self._num_test_samples = len(self.test_data["y"])

        return self._num_test_samples

    @property
    def num_samples(self):
        """Return the number of train + test samples for this client."""
        if not hasattr(self, "_num_samples"):
            self._num_samples = self.num_train_samples + self.num_test_samples

        return self._num_samples

    @property
    def train_sample_dist(self):
        """Return the distribution of train data for this client."""
        if not hasattr(self, "_train_sample_dist"):
            labels = self.train_data["y"]
            labels = labels.asnumpy().astype("int64")
            dist = np.bincount(labels)
            # align to num_classes
            num_classes = self.model.num_classes
            self._train_sample_dist = np.concatenate(
                (dist, np.zeros(num_classes - len(dist))))

        return self._train_sample_dist

    @property
    def next_train_batch_dist(self):
        """Return the distribution of next batch data for this client."""
        next_labels = self.next_batch_y.asnumpy().astype("int64")
        next_dist = np.bincount(next_labels)
        # align to num_classes
        num_classes = self.model.num_classes
        next_dist = np.concatenate(
            (next_dist, np.zeros(num_classes - len(next_dist))))
        return next_dist

    @property
    def test_sample_dist(self):
        """Return the distribution of test data for this client."""
        if not hasattr(self, "_test_sample_dist"):
            labels = self.test_data["y"]
            labels = labels.asnumpy().astype("int64")
            dist = np.bincount(labels)
            # align to num_classes
            num_classes = self.model.num_classes
            self._test_sample_dist = np.concatenate(
                (dist, np.zeros(num_classes - len(dist))))

        return self._test_sample_dist

    @property
    def sample_dist(self):
        """Return the distribution of overall data for this client."""
        if not hasattr(self, "_sample_dist"):
            self._sample_dist = self.train_sample_dist + self.test_sample_dist

        return self._sample_dist

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
