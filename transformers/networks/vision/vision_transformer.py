import pathlib

import einops as ein
import torch
import torch.nn as nn

from transformers.transformers import gen_positional_encoding, SelfAttentionTransformerEncoder


class VisionTransformer(nn.Module):
    def __init__(self, img_size: int, patch_size: int, num_classes: int, channels: int, num_layers: int, d_model: int,
                 n_heads: int, dropout: float):
        """
        Vision Transformer from "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (https://arxiv.org/abs/2010.11929)
        :param img_size: size of the image
        :param patch_size: size of the patch
        :param num_classes: number of classes
        :param channels: number of channels in the image
        :param num_layers: number of layers in the transformer
        :param d_model: dimension of the model
        :param n_heads: number of heads in the multi-head attention
        :param dropout: dropout rate
        """
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = channels * patch_size * patch_size

        # Patch projection
        self.patch_norm1 = nn.LayerNorm(self.patch_dim)
        self.patch_fc = nn.Linear(self.patch_dim, d_model)
        self.patch_norm2 = nn.LayerNorm(d_model)

        # Class token
        self.cls = nn.Parameter(torch.randn((1, 1, d_model)))
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Transformer
        self.trans_enc = SelfAttentionTransformerEncoder(num_layers, d_model, n_heads, dropout)

        # Final prediction layer
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        # Split image into patches and project them
        x = self._img_to_patch_embedding(x)

        # Add class token for classification
        cls = ein.repeat(self.cls, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat((cls, x), dim=1)

        # Add positional encoding
        num_tokens, dim = x.shape[1:]
        x += gen_positional_encoding(num_tokens, dim, x.device)

        # Apply dropout
        x = self.dropout(x)

        # Transformer self-attention
        x = self.trans_enc(x)

        # Generate prediction from the cls token
        x = x[:, 0]
        x = self.fc(self.norm(x))

        return x

    def _img_to_patch_embedding(self, img):
        """
        Split image into patches and project them
        :param img: image to split
        :return: Patches (batch_size, num_patches, patch_dim)
        """
        x = ein.rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.patch_norm1(x)
        x = self.patch_fc(x)
        x = self.patch_norm2(x)
        return x


if __name__ == '__main__':
    import poutyne as pt
    from transformers.utils.datasets import get_cifar10_dataloaders
    from transformers.utils.history import History

    # Training parameters
    epoch = 100
    batch_size = 256
    learning_rate = 1e-4
    train_split = 0.8

    # Model parameters
    img_size = 32
    patch_size = 8
    num_classes = 10
    channels = 3
    num_layers = 6
    dim_model = 512
    num_heads = 8
    dropout = 0.1

    # Data
    train_loader, valid_loader, test_loader = get_cifar10_dataloaders(train_split, batch_size)

    # Model and optimizer
    model = VisionTransformer(img_size, patch_size, num_classes, channels, num_layers, dim_model, num_heads, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model = pt.Model(model, optimizer, 'cross_entropy', batch_metrics=['accuracy'])
    model.cuda()

    # Training
    pathlib.Path('logs').mkdir(parents=True, exist_ok=True)
    history = model.fit_generator(train_loader, valid_loader, epochs=epoch, callbacks=[
        pt.ModelCheckpoint('logs/vit_best_epoch_{epoch}.ckpt', monitor='val_acc', mode='max', save_best_only=True,
                           keep_only_last_best=True, restore_best=True, verbose=True,
                           temporary_filename='best_epoch.ckpt.tmp'),
        pt.ReduceLROnPlateau(monitor='val_acc', mode='max', factor=0.5, patience=5, verbose=True),
    ])

    # Display training history
    History(history).display()

    # Test
    test_loss, test_acc = model.evaluate_generator(test_loader)
    print('test_loss: {:.4f} test_acc: {:.2f}'.format(test_loss, test_acc))
