from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn import Module, LayerNorm, Dropout, Linear, Sequential, ReLU
import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder, Softmax
import torch.nn.functional as F
from learned_cryptor.model_parts.decoder import FiLMTransformerDecoderLayer, FiLMTransformerDecoder
from learned_cryptor.model_parts.encoder import FiLMTransformerEncoderLayer, FiLMTransformerEncoder
from typing import Optional, Callable, Union, Any
import math

from learned_cryptor.model_parts.helper import FiLMBlock, _generate_square_subsequent_mask
from learned_cryptor.task import Task


class PositionalEncoding(torch.jit.ScriptModule):
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

    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

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


class FiLMTransformer(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, sentence_vocab_len: int, secret_vocab_len: int,
                 d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 embedding_dim: int = 64, src_contains_secret: bool = True,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        torch._C._log_api_usage_once(
            f"torch.nn.modules.{self.__class__.__name__}")

        self.sentence_vocab_len = sentence_vocab_len
        self.secret_vocab_len = secret_vocab_len
        self.src_contains_secret = src_contains_secret

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    bias,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = FiLMTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                        activation, layer_norm_eps, batch_first, norm_first,
                                                        bias, n_tasks=secret_vocab_len, embedding_dim=embedding_dim, src_contains_secret=src_contains_secret,
                                                        **factory_kwargs)
            decoder_norm = LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
            self.decoder = FiLMTransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm)

        self.sentence_embedding = torch.nn.Embedding(
            sentence_vocab_len, d_model)
        self._pos_encoding = PositionalEncoding(d_model, dropout)

        self._tokenizer = Linear(d_model, sentence_vocab_len)
        self._secret_tokenizer = Linear(d_model, secret_vocab_len)

        self._sentence_vocab_len = sentence_vocab_len
        self._secret_vocab_len = secret_vocab_len

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

        self._softmax = Softmax(dim=1)

    def forward(self, src: Tensor, secret: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                src_is_causal: Optional[bool] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> tuple[Tensor, Tensor]:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the Tensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the Tensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the Tensor mask for memory keys per batch (optional).
            src_is_causal: If specified, applies a causal mask as ``src_mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``src_is_causal`` provides a hint that ``src_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            tgt_is_causal: If specified, applies a causal mask as ``tgt_mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory_mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
              `(N, S, E)` if `batch_first=True`.
            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.
            - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.
            - tgt_mask: :math:`(T, T)` or :math:`(N\cdot\text{num\_heads}, T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decoder.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> # xdoctest: +SKIP
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        src_padding_mask = self.get_padding_mask(src)
        tgt_padding_mask = self.get_padding_mask(tgt)

        src = self.sentence_embedding(src) * math.sqrt(self.d_model)
        tgt = self.sentence_embedding(tgt) * math.sqrt(self.d_model)

        if not self.batch_first:
            src = src.transpose(0, 1)
            secret = secret.transpose(0, 1) if secret is not None else secret
            tgt = tgt.transpose(0, 1)

        src = self._pos_encoding(src)
        tgt = self._pos_encoding(tgt)

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model")

        tgt_mask = self.generate_square_subsequent_mask(
            tgt.shape[0], device=tgt.device, dtype=tgt.dtype)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask,
                              is_causal=src_is_causal)
        output = self.decoder(tgt, memory, secret, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)

        tokens_probs = self._tokenizer(output)
        tokens_probs = self._softmax(tokens_probs)
        return tokens_probs.permute(1, 2, 0), torch.argmax(tokens_probs, dim=-1).permute(1, 0)

    def get_padding_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tensor, dtype=torch.float).masked_fill(tensor == 2, float("-inf"))

    def encode(self, src: Tensor, secret: Tensor, task: Tensor, src_mask: Tensor):
        src = self.sentence_embedding(src)
        src = self._pos_encoding(src)
        src = torch.concat((src, secret), dim=1)

        return self.encoder(src, task, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, task: Tensor, tgt_mask: Tensor):
        if task.item() == Task.encode.value or task.item() == Task.decode_sentence.value:
            tgt = self.sentence_embedding(tgt)
        else:
            tgt = self.secret_embedding(tgt)
        tgt = self._pos_encoding(tgt)

        return self.decoder(tgt, memory, task, tgt_mask=tgt_mask)

    @staticmethod
    def generate_square_subsequent_mask(
            sz: int,
            device: torch.device = torch.device(
                torch._C._get_default_device()),  # torch.device('cpu'),
            dtype: torch.dtype = torch.get_default_dtype(),
    ) -> Tensor:
        r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return _generate_square_subsequent_mask(sz, dtype=dtype, device=device)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
