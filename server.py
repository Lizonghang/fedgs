from abc import ABC, abstractmethod
import numpy as np

from baseline_constants import BYTES_WRITTEN_KEY, \
    BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY


class Server(ABC):

    def __init__(self, server_model, merged_update):
        self.model = server_model
        self.merged_update = merged_update
        self.total_weight = 0

    @abstractmethod
    def select_clients(self, my_round, clients_per_group):
        return None

    @abstractmethod
    def train_model(self, num_syncs):
        return None

    def update_model(self):
        merged_update_ = list(self.merged_update.get_params())
        num_params = len(merged_update_)

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() / self.total_weight)

        self.model.set_params(self.merged_update.get_params())

        self.total_weight = 0
        self.merged_update.reset_zero()

    @abstractmethod
    def test_model(self, set_to_use):
        return None

    def save_model(self, log_dir):
        self.model.save(log_dir)


class TopServer(Server):

    def __init__(self, server_model, merged_update, servers):
        self.middle_servers = []
        self.register_middle_servers(servers)
        self.selected_clients = []
        super(TopServer, self).__init__(server_model, merged_update)

    def register_middle_servers(self, servers):
        if type(servers) == MiddleServer:
            servers = [servers]

        self.middle_servers.extend(servers)

    def select_clients(self, my_round, clients_per_group):
        selected_info = []
        self.selected_clients = []

        for s in self.middle_servers:
            _ = s.select_clients(my_round, clients_per_group)
            clients, info = _
            self.selected_clients.extend(clients)
            selected_info.extend(info)

        return selected_info

    def train_model(self, num_syncs):
        sys_metrics = {}

        for s in self.middle_servers:
            s.set_model(self.model)
            s_sys_metrics, update = s.train_model(num_syncs)
            self.merge_updates(s.num_selected_clients, update)

            sys_metrics.update(s_sys_metrics)

        self.update_model()

        return sys_metrics

    def merge_updates(self, num_clients, update):
        merged_update_ = list(self.merged_update.get_params())
        current_update_ = list(update)
        num_params = len(merged_update_)

        self.total_weight += num_clients

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() +
                (num_clients * current_update_[p].data()))

    def test_model(self, set_to_use="test"):
        metrics = {}

        for middle_server in self.middle_servers:
            middle_server.set_model(self.model)
            s_metrics = middle_server.test_model(set_to_use)
            metrics.update(s_metrics)

        return metrics


class MiddleServer(Server):

    def __init__(self, server_id, server_model, merged_update, clients_in_group):
        self.server_id = server_id
        self.clients = []
        self.register_clients(clients_in_group)
        self.selected_clients = []
        super(MiddleServer, self).__init__(server_model, merged_update)

    def register_clients(self, clients):
        if type(clients) is not list:
            clients = [clients]

        self.clients.extend(clients)

    def select_clients(self, my_round, clients_per_group):
        num_clients = min(clients_per_group, len(self.clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(
            self.online(self.clients), num_clients, replace=False)
        info = [(c.num_train_samples, c.num_test_samples)
                for c in self.selected_clients]
        return self.selected_clients, info

    def train_model(self, num_syncs):
        clients = self.selected_clients

        s_sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0}
            for c in clients}

        for s in range(num_syncs):
            for c in clients:
                c.set_model(self.model)
                comp, num_samples, update = c.train()
                self.merge_updates(num_samples, update)

                s_sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                s_sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
                s_sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] += comp

            self.update_model()

        update = self.model.get_params()
        return s_sys_metrics, update

    def merge_updates(self, client_samples, update):
        merged_update_ = list(self.merged_update.get_params())
        current_update_ = list(update)
        num_params = len(merged_update_)

        self.total_weight += client_samples

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() +
                (client_samples * current_update_[p].data()))

    def test_model(self, set_to_use="test"):
        s_metrics = {}

        for client in self.online(self.clients):
            client.set_model(self.model)
            c_metrics = client.test(set_to_use)
            s_metrics[client.id] = c_metrics

        return s_metrics

    def set_model(self, model):
        self.model.set_params(model.get_params())

    def online(self, clients):
        online_clients = clients
        assert len(online_clients) != 0, "No client available."
        return online_clients

    @property
    def num_clients(self):
        return len(self.clients)

    @property
    def num_selected_clients(self):
        return len(self.selected_clients)

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
              % (self.server_id, self.num_clients, self.num_samples,
                 self.num_train_samples, self.num_test_samples),
              file=log_fp, flush=True, end="")
        print("sample distribution:", list(self.sample_dist.astype("int64")))
