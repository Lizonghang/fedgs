import os
from mxnet import init, nd

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
        """Build and initialize the server model. If self.client_model is
        given, this ServerModel object is created for the server model,
        otherwise for the merged update. The server model will be synchronized
        with the client model, and the merged update will be initialized to zero.
        """
        self.net = build_net(
            self.dataset, self.model_name, self.num_classes, self.ctx)

        if self.client_model:
            self.set_params(self.client_model.get_params())
        else:
            self.net.initialize(
                init.Zero(), ctx=self.ctx, force_reinit=True)

    def reset_zero(self):
        """Reset the model data to zero, usually used to reset the merged update.
        Note that force reinit the model data with:
            self.net.initialize(
                init.Zero(), ctx=self.ctx, force_reinit=True)
        will leads to high cpu usage.
        """
        self.set_params([])

    def set_params(self, model_params):
        """Set the model data to the specified data. If an empty list is given,
        the model data will be set to zero.
        Args:
            model_params: The specified model data.
        """
        source_params = list(model_params)
        target_params = list(self.get_params())
        num_params = len(target_params)

        for p in range(num_params):
            if source_params:
                data = source_params[p].data()
            else:
                data = nd.zeros(target_params[p].shape, ctx=self.ctx)
            target_params[p].set_data(data)

    def get_params(self):
        """Return current model data.
        Returns:
            params: The current model data.
        """
        return self.net.collect_params().values()

    def save(self, log_dir):
        """Saves the server model to:
            {log_dir}/{model_name}.params
        """
        self.net.save_parameters(
            os.path.join(log_dir, self.model_name + ".params"))
