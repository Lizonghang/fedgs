import mxnet as mx
from mxnet.gluon import loss as gloss

from baseline_constants import ACCURACY_KEY, INPUT_SIZE
from model import Model
from utils.model_utils import build_net


class ClientModel(Model):
    def __init__(self, seed, dataset, model_name, ctx, lr, num_classes):
        self.dataset = dataset
        self.model_name = model_name
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr, ctx)

    def create_model(self):
        # Build a simple cnn network
        net = build_net(
            self.dataset, self.model_name, self.num_classes, self.ctx, self.seed)

        # Use softmax cross-entropy loss
        loss = gloss.SoftmaxCrossEntropyLoss()

        # Create trainer
        trainer = mx.gluon.Trainer(
            params=net.collect_params(),
            optimizer=self.optimizer,
            optimizer_params={"learning_rate": self.lr}
        )

        return net, loss, trainer

    def test(self, data):
        # Process train data and labels before inference
        x_vecs = self.preprocess_x(data["x"])
        labels = self.preprocess_y(data["y"])

        # Model inference
        output = self.net(x_vecs)

        # Calculate accuracy and loss
        acc = (output.argmax(axis=1) == labels).mean().asscalar()
        loss = self.loss(output, labels).mean().asscalar()
        return {ACCURACY_KEY: acc, "loss": loss}

    def preprocess_x(self, raw_x_batch):
        return raw_x_batch.reshape((-1, *INPUT_SIZE))

    def preprocess_y(self, raw_y_batch):
        return raw_y_batch
