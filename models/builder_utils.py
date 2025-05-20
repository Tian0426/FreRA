import torch
from torch import nn
# from .MMB import *

class Classifier(nn.Module):
    def __init__(self, bb_dim, n_classes):
        super(Classifier, self).__init__()

        self.classifier = nn.Linear(bb_dim, n_classes)

    def forward(self, x):
        out = self.classifier(x)

        return out


class Projector(nn.Module):
    def __init__(self, model, bb_dim, prev_dim, dim):
        super(Projector, self).__init__()
        if model == 'SimCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim))
        elif model == 'byol':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim, affine=False))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.projector(x)
        return x


class Predictor(nn.Module):
    def __init__(self, model, dim, pred_dim):
        super(Predictor, self).__init__()
        if model == 'SimCLR':
            pass
        elif model == 'byol':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.predictor(x)
        return x

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


from functools import wraps


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projector and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, DEVICE, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.DEVICE = DEVICE

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        children = [*self.net.children()]
        print('children[self.layer]:', children[self.layer])
        return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.reshape(output.shape[0], -1)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = Projector(model='byol', bb_dim=dim, prev_dim=self.projection_hidden_size, dim=self.projection_size)
        return projector.to(hidden)

    def get_representation(self, x):

        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)

        if len(representation.shape) == 3:
            representation = representation.reshape(representation.shape[0], -1)

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

