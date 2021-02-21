from abc import ABC, abstractmethod
import scipy
import numpy as np

from baseline_constants import BYTES_WRITTEN_KEY, \
    BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY


class Server(ABC):

    def __init__(self, server_model, merged_update):
        self.model = server_model
        self.merged_update = merged_update
        self.total_weight = 0

    @abstractmethod
    def select_clients(
            self, my_round, clients_per_group, sample, base_dist, display, metrics_dir):
        """Select clients_per_group clients from each group.
        Args:
            my_round: The current training round, used for
                random sampling.
            clients_per_group: Number of clients to select in
                each group.
            sample: Sample method, either "random" or "approx_iid".
            base_dist: Real data distribution, usually global_test_dist.
            display: Visualize data distribution when set to True.
            metrics_dir: Directory to save metrics files.
        Returns:
            selected_clients: List of clients being selected.
            client_info: List of (num_train_samples, num_test_samples)
                of selected clients.
        """
        return None

    @abstractmethod
    def train_model(self, num_syncs):
        """Aggregate clients' models after each iteration. If
        num_syncs synchronizations are reached, middle servers'
        models are then aggregated at the top server.
        Args:
            num_syncs: Number of client - middle server synchronizations
                in each round before sending to the top server.
        Returns:
            metrics: Dict of metrics returned by the model.
            update: The model after training num_syncs synchronizations.
        """
        return None

    def update_model(self):
        """Update self.model with averaged merged update."""
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
        """Test self.model on all clients.
        Args:
            set_to_use: Dataset to test on, either "train" or "test".
        Returns:
            metrics: Dict of metrics returned by the model.
        """
        return None

    def save_model(self, log_dir):
        """Save self.model to specified directory.
        Args:
            log_dir: Directory to save model file.
        """
        self.model.save(log_dir)


class TopServer(Server):

    def __init__(self, server_model, merged_update, servers):
        self.middle_servers = []
        self.register_middle_servers(servers)
        self.selected_clients = []
        super(TopServer, self).__init__(server_model, merged_update)

    def register_middle_servers(self, servers):
        """Register middle servers.
        Args:
            servers: Middle servers to be registered.
        """
        if type(servers) == MiddleServer:
            servers = [servers]

        self.middle_servers.extend(servers)

    def select_clients(self, my_round, clients_per_group, sample="random",
                       base_dist=None, display=False, metrics_dir="metrics"):
        """Call middle servers to select clients."""
        assert sample in ["random", "approx_iid"]

        selected_info = []
        self.selected_clients = []

        for s in self.middle_servers:
            _ = s.select_clients(
                my_round, clients_per_group, sample, base_dist, display, metrics_dir)
            clients, info = _
            self.selected_clients.extend(clients)
            selected_info.extend(info)

        return selected_info

    def train_model(self, num_syncs):
        """Call middle servers to train their models and aggregate
        their updates."""
        sys_metrics = {}

        for s in self.middle_servers:
            s.set_model(self.model)
            s_sys_metrics, update = s.train_model(num_syncs)
            self.merge_updates(s.num_selected_clients, update)

            sys_metrics.update(s_sys_metrics)

        self.update_model()

        return sys_metrics

    def merge_updates(self, num_clients, update):
        """Aggregate updates from middle servers based on the
        number of selected clients.
        Args:
            num_clients: Number of selected clients for this
                middle server.
            update: The model trained by this middle server.
        """
        merged_update_ = list(self.merged_update.get_params())
        current_update_ = list(update)
        num_params = len(merged_update_)

        self.total_weight += num_clients

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() +
                (num_clients * current_update_[p].data()))

    def test_model(self, set_to_use="test"):
        """Call middle servers to test their models."""
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
        """Register clients of this middle server.
        Args:
            clients: Clients to be registered.
        """
        if type(clients) is not list:
            clients = [clients]

        self.clients.extend(clients)

    def select_clients(self, my_round, clients_per_group, sample="random",
                       base_dist=None, display=False, metrics_dir="metrics"):
        """Randomly select clients_per_group clients for this round."""
        online_clients = self.online(self.clients)
        num_clients = len(online_clients)
        num_sample_clients = min(clients_per_group, num_clients)
        num_rand_clients = num_sample_clients // 2
        num_best_clients = num_sample_clients - num_rand_clients

        # Randomly select half of num_clients clients
        np.random.seed(my_round)
        rand_clients_idx = np.random.choice(
            range(num_clients), num_rand_clients, replace=False)
        rand_clients = np.take(online_clients, rand_clients_idx)

        # Select rest clients to meet approximate i.i.d. dist
        rest_clients = np.delete(online_clients, rand_clients_idx)
        if sample == "random":
            best_clients = self.random_sampling(
                rest_clients, num_best_clients)
        elif sample == "approx_iid":
            best_clients = self.approximate_iid_sampling(
                rest_clients, num_best_clients, base_dist)

        self.selected_clients = np.concatenate([rand_clients, best_clients])

        # Measure the distance of base distribution and mean distribution
        distance = self.get_dist_distance(
            self.selected_clients, base_dist, use_distance="wasserstein")
        print("Dist Distance on Middle Server %i:" % self.server_id, distance)

        # Visualize distributions if needed
        if display:
            from metrics.visualization_utils import plot_clients_dist

            plot_clients_dist(clients=self.selected_clients,
                              global_test_dist=base_dist,
                              draw_mean=True,
                              metrics_dir=metrics_dir)

        info = [(c.num_train_samples, c.num_test_samples)
                for c in self.selected_clients]
        return self.selected_clients, info

    def random_sampling(self, clients, num_clients):
        """Randomly sample num_clients clients from given clients.
        Args:
            clients: List of clients to be sampled.
            num_clients: Number of clients to sample.
        Returns:
            rand_clients: List of randomly sampled clients.
        """
        return np.random.choice(clients, num_clients, replace=False)

    def approximate_iid_sampling(self, clients, num_clients, base_dist):
        """TODO(Yihong): Implement approximate i.i.d. sampling algorithm.
        Args:
            clients: List of clients to be sampled.
            num_clients: Number of clients to sample.
            base_dist: Real data distribution, usually global_test_dist.
        Returns:
            best_clients: List of sampled clients, which makes
                self.selected_clients satisfy i.i.d. distribution.
        """
        return np.array([])

    def get_dist_distance(self, clients, base_dist, use_distance="wasserstein"):
        """Return distance of the base distribution and the mean distribution.
        Args:
            clients: List of sampled clients.
            base_dist: Real data distribution, usually global_test_dist.
            use_distance: Distance metric to be used, could be:
                ["l1", "l2", "cosine", "js", "wasserstein"].
        Returns:
            distance: The L1 distance of the base distribution and the mean
                distribution.
        """
        c_mean_dist_ = sum([c.train_sample_dist for c in clients])
        c_mean_dist_ = c_mean_dist_ / c_mean_dist_.sum()
        base_dist_ = base_dist / base_dist.sum()

        distance = np.inf
        if use_distance == "l1":
            dist_diff_ = c_mean_dist_ - base_dist_
            distance = np.linalg.norm(dist_diff_, ord=1)
        elif use_distance == "l2":
            dist_diff_ = c_mean_dist_ - base_dist_
            distance = np.linalg.norm(dist_diff_, ord=2)
        elif use_distance == "cosine":
            # The cosine distance between vectors u and v is defined as:
            #       1 - dot(u, v) / (norm(u, ord=2) * norm(v, ord=2))
            distance = scipy.spatial.distance.cosine(c_mean_dist_, base_dist_)
        elif use_distance == "js":
            distance = scipy.spatial.distance.jensenshannon(c_mean_dist_, base_dist_)
        elif use_distance == "wasserstein":
            distance = scipy.stats.wasserstein_distance(c_mean_dist_, base_dist_)

        return distance

    def train_model(self, num_syncs):
        """Train self.model for num_syncs synchronizations."""
        clients = self.selected_clients

        s_sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0}
            for c in clients}

        for _ in range(num_syncs):
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
        """Aggregate updates from clients based on the size
        of batched data.
        Args:
            client_samples: Size of batched data used by this client.
            update: The model trained by this client.
        """
        merged_update_ = list(self.merged_update.get_params())
        current_update_ = list(update)
        num_params = len(merged_update_)

        self.total_weight += client_samples

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() +
                (client_samples * current_update_[p].data()))

    def test_model(self, set_to_use="test"):
        """Test self.model on online clients."""
        s_metrics = {}

        for client in self.online(self.clients):
            client.set_model(self.model)
            c_metrics = client.test(set_to_use)
            s_metrics[client.id] = c_metrics

        return s_metrics

    def set_model(self, model):
        """Set the model data to specified model.
        Args:
            model: The specified model.
        """
        self.model.set_params(model.get_params())

    def online(self, clients):
        """Return clients that are online.
        Args:
            clients: List of all clients registered at this
                middle server.
        Returns:
            online_clients: List of all online clients.
        """
        online_clients = clients
        assert len(online_clients) != 0, "No client available."
        return online_clients

    @property
    def num_clients(self):
        """Return the number of all clients registered at this
        middle server."""
        return len(self.clients)

    @property
    def num_selected_clients(self):
        """Return the number of selected clients."""
        return len(self.selected_clients)

    @property
    def num_samples(self):
        """Return the total number of samples for self.clients."""
        return sum([c.num_samples for c in self.clients])

    @property
    def num_train_samples(self):
        """Return the total number of train samples for
        self.clients."""
        return sum([c.num_train_samples for c in self.clients])

    @property
    def num_test_samples(self):
        """Return the total number of test samples for
        self.clients."""
        return sum([c.num_test_samples for c in self.clients])

    @property
    def train_sample_dist(self):
        """Return the distribution of train data for
        self.clients."""
        return sum([c.train_sample_dist for c in self.clients])

    @property
    def test_sample_dist(self):
        """Return the distribution of test data for
        self.clients."""
        return sum([c.test_sample_dist for c in self.clients])

    @property
    def sample_dist(self):
        """Return the distribution of overall data for
        self.clients."""
        return self.train_sample_dist + self.test_sample_dist

    def brief(self, log_fp):
        """Briefly summarize the statistics of this middle server"""
        print("[Group %i] Number of clients: %i, number of samples: %i, "
              "number of train samples: %s, number of test samples: %i, "
              % (self.server_id, self.num_clients, self.num_samples,
                 self.num_train_samples, self.num_test_samples),
              file=log_fp, flush=True, end="")
        print("sample distribution:", list(self.sample_dist.astype("int64")))
