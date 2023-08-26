# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, enc_dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        self.enc_dim = enc_dim
        self.pred_dim = pred_dim

        # build a 3-layer projector
        projector = nn.Sequential(
                        nn.Linear(base_encoder.embed_dim, base_encoder.embed_dim, bias=False),
                        nn.BatchNorm1d(base_encoder.embed_dim),
                        nn.ReLU(inplace=True), # first layer
                        nn.Linear(base_encoder.embed_dim, base_encoder.embed_dim, bias=False),
                        nn.BatchNorm1d(base_encoder.embed_dim),
                        nn.ReLU(inplace=True), # second layer
                        nn.Linear(base_encoder.embed_dim, self.enc_dim, bias=False),
                        nn.BatchNorm1d(self.enc_dim, affine=False)
                    )
        self.encoder = nn.Sequential(base_encoder, projector)

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
                            nn.Linear(self.enc_dim, self.pred_dim, bias=False),
                            nn.BatchNorm1d(self.pred_dim),
                            nn.ReLU(inplace=True), 
                            nn.Linear(self.pred_dim, self.enc_dim)
                        )

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
    
    
class Block(nn.Module):
    def __init__(self, hidden_features, drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, residual=True):
        super().__init__()
        self.residual = residual
        self.norm = norm_layer(hidden_features)
        self.fc = nn.Linear(hidden_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.residual:
            return x + self.drop(self.act(self.fc(self.norm(x))))
        else:
            return self.drop(self.act(self.fc(self.norm(x))))
        

class VisionMLP(nn.Module):
    """ Vision MLP """
    def __init__(self, input_size, depth, hidden_features, drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, residual=True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.first_layer = nn.Linear(input_size, hidden_features)
        self.blocks = nn.ModuleList([Block(hidden_features, drop, act_layer, norm_layer, residual) for i in range(depth)])
        self.norm = norm_layer(hidden_features)
        self.embed_dim = hidden_features

    def forward(self, x):
        x = self.first_layer(self.flatten(x))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


def vimlp_huge(**kwargs):
    model = VisionMLP(input_size=224*224*3, depth=16, hidden_features=13125, **kwargs)
    return model