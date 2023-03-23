import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.transformers import SelfAttentionTransformerEncoder, positional_encoding, FeedForward


class ChatLite(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nlayers: int, nhead: int, dropout: float):
        """
        Simple character-level language model
        :param vocab_size: number of characters in the vocabulary
        :param d_model: dimension of the model
        :param nlayers: number of layers in the transformer
        :param nhead: number of heads in the multi-head attention
        :param dropout: dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = SelfAttentionTransformerEncoder(nlayers, d_model, nhead, dropout)
        self.ff = FeedForward(d_model, d_model * 4, vocab_size, dropout)

    def forward(self, x):
        """
        Predict the next characters
        :param x: input characters
        :return: predicted logits
        """
        # Embed the characters
        x = self.embedding(x)
        # Add positional encoding
        _, num_tokens, d_model = x.shape
        x += positional_encoding(num_tokens, d_model, x.device)
        # Self-attention
        x = self.transformer(x, is_causal=True)
        # Final feed-forward layer
        x = self.ff(x)
        return x

    def loss(self, pred, y):
        """
        Compute the loss
        :param pred: predicted logits
        :param y: true labels
        :return: loss
        """
        _, num_tokens, d_model = pred.shape
        # Flatten the predictions and the labels
        pred = pred.view(-1, d_model)
        y = y.view(-1)
        # Compute the loss
        return F.cross_entropy(pred, y)

    @torch.no_grad()
    def generate(self, idx: int, max_new_tokens: int, block_size: int):
        """
        Generate new tokens
        :param idx: context tokens
        :param max_new_tokens: maximum number of new tokens to generate
        :param block_size: size of the context
        :return: generated tokens
        """
        model.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        model.train()
        return idx


if __name__ == '__main__':
    import os
    import pathlib
    import matplotlib.pyplot as plt

    # Download the data
    data_path = pathlib.Path('data/tiny_shakespeare.txt')
    if not data_path.exists():
        os.system(
            f'curl -o {str(data_path)} https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt')

    # Read the data
    with open(data_path, 'r') as f:
        text = f.read()

    # Compute stats on the data
    characters = sorted(list(set(text)))
    vocab_size = len(characters)

    str_to_int = {c: i for i, c in enumerate(characters)}
    int_to_str = {v: k for k, v in str_to_int.items()}


    def encode(s):
        """
        Encode a string into a list of integers
        :param s: string
        :return: encoded string
        """
        return [str_to_int[c] for c in s]


    def decode(s):
        """
        Decode a list of integers into a string
        :param s: encoded string
        :return: string
        """
        return [int_to_str[c] for c in s]


    # Data
    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]


    def get_batch(split, block_size, device):
        """
        Get a random batch of data
        :param split: train or val
        :param block_size: context size
        :param device: device
        :return: (x, y) where x is the context and y is the target
        """
        data = train_data if split == 'train' else val_data
        indices = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in indices])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in indices])
        x, y = x.to(device), y.to(device)
        return x, y


    # Training parameters
    batch_size = 128
    block_size = 256
    max_iters = 10_000
    eval_interval = 1_000
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200

    # Model parameters
    d_model = 384
    nhead = 6
    nlayer = 6
    dropout = 0.1

    # Model and optimizer
    model = ChatLite(vocab_size, d_model, nlayer, nhead, dropout)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    iter_loss = []
    for i in range(max_iters):
        # Evaluate the model
        if i % eval_interval == 0 or i == max_iters - 1:
            with torch.no_grad():
                out = {}
                model.eval()
                for split in ['train', 'val']:
                    losses = torch.zeros(eval_iters)
                    for k in range(eval_iters):
                        x, y = get_batch(split, block_size, device)
                        pred = model(x)
                        loss = model.loss(pred, y)
                        losses[k] = loss.item()
                    out[split] = losses.mean()
                print(f"step {i}: train loss {out['train'].item():.4f}, val loss {out['val'].item():.4f}")
                iter_loss.append((out['train'].item(), out['val'].item()))
                model.train()
        # Train the model
        x, y = get_batch('train', block_size, device)

        pred = model(x)
        loss = model.loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Display the loss
    iter_loss_np = np.array(iter_loss)
    plt.plot(iter_loss_np[:, 0], label='train')
    plt.plot(iter_loss_np[:, 1], label='valid')
    plt.legend()
    plt.show()

    # Generate some text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(''.join(decode(model.generate(context, 500, block_size)[0].tolist())))
