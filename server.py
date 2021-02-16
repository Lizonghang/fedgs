from abc import ABC, abstractmethod
import os
import numpy as np
from mxnet import init

from utils.model_utils import build_net
from baseline_constants import BYTES_WRITTEN_KEY, \
    BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY


class Server(ABC):

    def __init__(self, server_model, merged_update):
        self.model = server_model
        self.merged_update = merged_update
        self.total_weight = 0

    @abstractmethod
    def test_model(self, clients_to_test, set_to_use):
        return None


class TopServer(Server):

    def __init__(self, server_model, merged_update):
        super(TopServer, self).__init__(server_model, merged_update)

    def test_model(self, clients_to_test, set_to_use="test"):
        """Tests self.model on given clients.
        Tests model on self.selected_clients if clients_to_test=None.
        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        return None


class MidServer(Server):

    def __init__(self, group_id, server_model, merged_update, clients_in_group):
        self.group_id = group_id
        self.clients = clients_in_group
        super(MidServer, self).__init__(server_model, merged_update)

    def test_model(self, clients_to_test, set_to_use="test"):
        return None

    @property
    def num_clients(self):
        return len(self.clients)

    @property
    def num_samples(self):
        return sum([c.num_samples for c in self.clients])

    @property
    def num_train_samples(self):
        return sum([c.num_train_samples for c in self.clients])

    @property
    def num_test_samples(self):
        return sum([c.num_test_samples for c in self.clients])

    @property
    def train_sample_dist(self):
        return sum([c.train_sample_dist for c in self.clients])

    @property
    def test_sample_dist(self):
        return sum([c.test_sample_dist for c in self.clients])

    @property
    def sample_dist(self):
        return self.train_sample_dist + self.test_sample_dist

    def brief(self, log_fp):
        print("[Group %i] Number of clients: %i, number of samples: %i, "
              "number of train samples: %s, number of test samples: %i, "
              % (self.group_id, self.num_clients, self.num_samples,
                 self.num_train_samples, self.num_test_samples),
              file=log_fp, flush=True, end="")
        print("sample distribution:", list(self.sample_dist.astype("int64")))
