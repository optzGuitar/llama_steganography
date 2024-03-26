import math
import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        max_len *= 2

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe.repeat((x.shape[0], 1, 1))[:, :x.shape[1], :]
        return self.dropout(x)


class SeperateModel(nn.Module):
    def __init__(self, n_tokens, n_tokens_out, d_model, device):
        super().__init__()

        self._embedding = nn.Embedding(n_tokens, d_model, device=device)
        self._tokenizer = nn.Linear(d_model, n_tokens_out, device=device)
        self._model = nn.Transformer(d_model=d_model)

        self._pos_encoding = PositionalEncoding(d_model)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, src, tgt):
        src = self._embedding(src)
        tgt = self._embedding(tgt)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        src = self._pos_encoding(src)
        tgt = self._pos_encoding(tgt)

        x = self._model(src, tgt)
        x = self._tokenizer(x)

        x = self._softmax(x)

        return x.permute(1, 2, 0), x.argmax(dim=2).permute(1, 0)


class SeperateEncoder(nn.Module):
    def __init__(self, n_tokens, n_secret_tokens, d_model, device):
        super().__init__()

        self._embedding = nn.Embedding(n_tokens, d_model, device=device)
        self._secret_embedding = nn.Embedding(
            n_secret_tokens, d_model, device=device
        )
        self._tokenizer = nn.Linear(d_model, n_tokens, device=device)
        self._model = nn.Transformer(d_model=d_model)

        self._pos_encoding = PositionalEncoding(d_model)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, src, secret, tgt):
        src = self._embedding(src)
        secret = self._secret_embedding(secret)
        tgt = self._embedding(tgt)
        src = torch.cat((src, secret), dim=1)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        src = self._pos_encoding(src)
        tgt = self._pos_encoding(tgt)

        x = self._model(src, tgt)
        x = self._tokenizer(x)

        x = self._softmax(x)

        return x.permute(1, 2, 0), x.argmax(dim=2).permute(1, 0)
