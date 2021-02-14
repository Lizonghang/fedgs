from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock


class CNN(HybridBlock):

    def __init__(self, num_classes, **kwargs):
        super(CNN, self).__init__(**kwargs)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            with self.features.name_scope():

                self.features.add(nn.Conv2D(
                    channels=32,
                    kernel_size=5,
                    padding=2,
                    activation="relu")
                )
                self.features.add(nn.MaxPool2D(
                    pool_size=2,
                    strides=2
                ))
                self.features.add(nn.Conv2D(
                    channels=64,
                    kernel_size=5,
                    padding=2,
                    activation="relu"
                ))
                self.features.add(nn.MaxPool2D(
                    pool_size=2,
                    strides=2
                ))
                self.features.add(nn.Flatten())
                self.features.add(nn.Dense(
                    units=2048,
                    activation="relu"
                ))

            self.output = nn.Dense(units=num_classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x

    def set_params(self, model_params):
        source_params = list(model_params)
        target_params = list(self.get_params())
        num_params = len(target_params)
        for p in range(num_params):
            target_params[p].set_data(source_params[p].data())

    def get_params(self):
        return self.collect_params().values()

    def brief(self):
        return list(self.get_params())[0].data()[0, 0, :, :]


def build_net(num_classes, **kwargs):
    net = CNN(num_classes, **kwargs)
    return net
