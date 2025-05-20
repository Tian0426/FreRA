import torch
import torch.nn as nn
from random import sample
from models.backbones import FCN
from models.NTXent import *
from models.builder_utils import *
import copy

class SimCLR(nn.Module):
    def __init__(self, device, dataset, n_feature, batch_size, base_encoder, dim=128, T=0.1):
        super(SimCLR, self).__init__()

        if base_encoder == 'FCN':
            self.encoder_q = FCN(dataset, n_channels=n_feature, n_classes=dim, backbone=False)
        dim_mlp = self.encoder_q.logits.weight.shape[1]
        self.encoder_q.logits = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.logits)



        self.NTXentLoss = NTXentLoss(device=device, batch_size=batch_size, temperature=T)

    def forward(self, im_q, im_k):
        z1 = self.encoder_q(im_q)
        z2 = self.encoder_q(im_k)

        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        logits, labels = self.NTXentLoss(z1, z2)

        return logits, labels, z1, z2

class BYOL(nn.Module):
    def __init__(
        self,
        DEVICE,
        base_encoder,
        dataset,
        n_feature,
        window_size,
        hidden_layer = -1,
        projection_size = 128,
        moving_average = 0.99,
        use_momentum = True,
    ):
        super().__init__()

        if base_encoder == 'FCN':
            self.encoder_q = FCN(dataset, n_channels=n_feature, n_classes=projection_size, backbone=False)

        dim_mlp = self.encoder_q.logits.weight.shape[1]
        self.encoder_q = NetWrapper(self.encoder_q, projection_size, dim_mlp, DEVICE=DEVICE, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average)

        self.online_predictor = Predictor(model='byol', dim=projection_size, pred_dim=projection_size)

        self.to(DEVICE)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, window_size, n_feature, device=DEVICE),
                     torch.randn(2, window_size, n_feature, device=DEVICE))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.encoder_q)
        for p in target_encoder.parameters():
            p.requires_grad = False
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.encoder_q)

    def forward(
        self,
        im_q,
        im_k,
    ):
        assert not (self.training and im_q.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        online_proj_one, lat1 = self.encoder_q(im_q)
        online_proj_two, lat2 = self.encoder_q(im_k)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.encoder_q
            target_proj_one, _ = target_encoder(im_q)
            target_proj_two, _ = target_encoder(im_k)
            target_proj_one.detach_()
            target_proj_two.detach_()

        return online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach()
