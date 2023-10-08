import pathlib

import einops as ein
import torch
import torch.nn as nn

from nnet.positional_encoding.absolute_positional_encoding import absolute_positional_encoding
from nnet.transformers import TransformerEncoder, SelfAttentionTransformerEncoder


class PerceiverIO(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_head: int, n_latent: int, n_layer: int, n_output: int, out_dim: int,
                 dropout: float):
        """
        Perceiver from "Perceiver: General Perception with Iterative Attention" (Jaegle, 2021)
        :param in_dim: dimension of the input array
        :param d_model: dimension of the model
        :param n_head: number of heads in the multi-head attention
        :param n_layer: number of self-attention layers
        :param n_latent: number of latent variables
        :param n_output: number of output variables
        :param out_dim: dimension of the projection of output array
        :param dropout: dropout rate
        """
        super().__init__()
        self.input_array_proj = nn.Linear(in_dim, d_model)
        self.latent_array = nn.Parameter(torch.randn((1, n_latent, d_model)))
        self.output_array = nn.Parameter(torch.randn((1, n_output, d_model)))
        self.cross_attention = TransformerEncoder(1, d_model, n_head, dropout)
        self.self_attention = SelfAttentionTransformerEncoder(n_layer, d_model, n_head, dropout)
        self.output_transformer = TransformerEncoder(1, d_model, n_head, dropout)
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x):
        """
        Forward pass of PerceiverIO
        :param x: input array of shape (batch_size, n, dim)
        :return: PerceiverIO output of shape (batch_size, n_output, out_dim)
        """
        batch_size = x.shape[0]

        # Repeat latent and output arrays for batch
        latent = ein.repeat(self.latent_array, '1 n d -> b n d', b=batch_size)
        output = ein.repeat(self.output_array, '1 n d -> b n d', b=batch_size)

        # Project input array to make it of shape (batch_size, n, d_model)
        x = self.input_array_proj(x)

        # Add position encoding
        num_tokens, dim = x.shape[1:]
        x += absolute_positional_encoding(num_tokens, dim, device=x.device)

        # Apply cross-attention from latent to input array
        latent = self.cross_attention(q=latent, k=x, v=x)

        # Apply self-attention to latent array
        latent = self.self_attention(latent)

        # Apply cross-attention from output to latent array
        output = self.output_transformer(q=output, k=latent, v=latent)

        # Final projection of output array
        output = self.fc(output)

        return output


class ClassificationPerceiverIO(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_head: int, n_latent: int, n_layer: int,
                 n_output: int, out_dim: int, num_classes: int, dropout: float, pooling: str = 'mean'):
        """
        Classification using PerceiverIO
        :param in_dim: dimension of the input array
        :param d_model: dimension of the model
        :param n_head: number of heads in the multi-head attention
        :param n_latent: number of latent variables
        :param n_layer: number of self-attention layers
        :param n_output: number of output variables
        :param out_dim: dimension of the projection of output array, output of the PerceiverIO
        :param num_classes: number of classes
        :param dropout: dropout rate
        :param pooling: pooling method to use, either 'mean', 'add' or 'max'
        """
        super().__init__()
        self.pooling = pooling
        self.perceiver_io = PerceiverIO(in_dim, d_model, n_head, n_latent, n_layer, n_output, out_dim, dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        # Flatten the input array to make it of shape (batch_size, in_dim, n), n being the number of pixels
        x = x.flatten(start_dim=2)

        # Rearrange the input array to make it of shape (batch_size, n, in_dim)
        # so that the second dimension is the number of tokens
        x = ein.rearrange(x, 'b c n -> b n c')

        # Apply PerceiverIO
        x = self.perceiver_io(x)

        # Pooling to get a single vector
        x = self._pooling_out(x)

        # Final projection
        x = self.fc(x)

        return x

    def _pooling_out(self, x):
        if self.pooling == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.pooling == 'mean':
            x = torch.mean(x, dim=1)
        elif self.pooling == 'add':
            x = torch.sum(x, dim=1)
        return x


if __name__ == '__main__':
    import poutyne as pt
    from nnet.utils.datasets import get_cifar10_dataloaders
    from nnet.utils.history import History

    # Training parameters
    epoch = 100
    batch_size = 128
    learning_rate = 1e-4
    train_split = 0.8
    device = 'cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else 'cpu')

    # Model parameters
    in_dim = 3
    d_model = 512
    n_head = 8
    n_latent = 128
    n_layer = 6
    n_output = 128
    out_dim = 512
    num_classes = 10
    dropout = 0.1
    pooling = 'mean'

    # Data
    train_loader, valid_loader, test_loader = get_cifar10_dataloaders(train_split, batch_size, num_workers=8)

    # Model and optimizer
    model = ClassificationPerceiverIO(in_dim, d_model, n_head, n_latent, n_layer, n_output, out_dim,
                                      num_classes, dropout, pooling)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model = pt.Model(model, optimizer, 'cross_entropy', batch_metrics=['accuracy'])
    model.to(device)

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
