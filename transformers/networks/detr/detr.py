import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

import einops as ein

from transformers.transformers import Transformer
from transformers.utils.datasets.detection.toy_detection_dataset import ToyDetectionDataset


class DETR(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, max_detections: int, d_model: int, n_head: int,
                 num_layers_enc: int, num_layers_dec: int, dropout: float = 0.1):
        """
        DETR from "End-to-End Object Detection with Transformers" (Carion, 2020)
        :param backbone: backbone
               for example `nn.Sequential(*list(resnet50(pretrained=True).children())[:-2], nn.Conv2d(2048, d_model, 1))`
        :param num_classes: number of classes
        :param max_detections: maximum number of detections
        :param d_model: dimension of the model
        :param n_head: number of heads in the multi-head attention
        :param num_layers_enc: number of encoder layers
        :param num_layers_dec: number of decoder layers
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = Transformer(num_layers_enc, num_layers_dec, d_model, n_head, None, dropout)
        self.lin_class = nn.Linear(d_model, num_classes + 1)
        self.lin_bbox = nn.Linear(d_model, 4)
        self.query_pos = nn.Parameter(torch.rand(1, max_detections, d_model))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x).flatten(2).transpose(1, 2)
        query_pos = ein.repeat(self.query_pos, '1 m d -> b m d', b=batch_size)
        h = self.transformer(x, query_pos)
        return self.lin_class(h), self.lin_bbox(h).sigmoid()


if __name__ == '__main__':
    import pathlib

    import poutyne as pt
    from transformers.utils.history import History

    # Training parameters
    epoch = 100
    batch_size = 128
    learning_rate = 1e-4
    train_split = 0.8
    val_split = 0.1

    # Model parameters
    in_dim = 3
    d_model = 96
    n_head = 8
    n_latent = 128
    n_layer = 6
    n_output = 128
    out_dim = 512
    num_classes = 10
    dropout = 0.1
    pooling = 'mean'
    num_workers = 0

    # Data
    dataset = ToyDetectionDataset(10, 1, 10_000)

    train_idx = np.arange(math.floor(train_split * len(dataset)))
    valid_idx = np.arange(train_split * len(dataset), math.floor((train_split + val_split) * len(dataset)))
    test_idx = np.arange(math.floor((train_split + val_split) * len(dataset)), len(dataset))

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)

    # Model and optimizer
    model = DETR(nn.Sequential(nn.Conv2d(1, d_model, 3)), 1, 5, d_model, n_head, n_layer, n_layer, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model = pt.Model(model, optimizer, 'cross_entropy', batch_metrics=['accuracy'])
    model.cuda()

    # Training
    pathlib.Path('logs').mkdir(parents=True, exist_ok=True)
    history = model.fit_generator(train_loader, valid_loader, epochs=epoch, callbacks=[
        pt.ModelCheckpoint('logs/perceiver_io_best_epoch_{epoch}.ckpt', monitor='val_acc', mode='max',
                           save_best_only=True,
                           keep_only_last_best=True, restore_best=True, verbose=True,
                           temporary_filename='best_epoch.ckpt.tmp'),
        pt.ReduceLROnPlateau(monitor='val_acc', mode='max', factor=0.5, patience=5, verbose=True),
    ])

    # Display training history
    History(history).display()

    # Test
    test_loss, test_acc = model.evaluate_generator(test_loader)
    print('test_loss: {:.4f} test_acc: {:.2f}'.format(test_loss, test_acc))
