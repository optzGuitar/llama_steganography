import copy
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional
from torch.nn import ModuleList
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, embed_len: int, shape: tuple[int, ...], device) -> None:
        super().__init__()
        self.embed_len = embed_len
        self.shape = shape

        stride = 3
        kernel = 5
        padding_height = (stride * (shape[0] - 1) + 5 - shape[0]) // 2
        padding_width = (stride * (shape[1] - 1) + 5 - shape[1]) // 2

        self.linear = nn.Linear(embed_len, shape[0] * shape[1], device=device)
        # self.conf = nn.Conv2d(1, 1, kernel_size=kernel, padding=(
        #     padding_height, padding_width), stride=stride, device=device)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        # x = x.view(x.shape[0], 1, *self.shape)
        # x = self.relu(x)
        # x = self.conf(x)
        x = self.tanh(x)
        return x.view(x.shape[0], *self.shape)


class FiLMBlock(nn.Module):
    def __init__(self, task_embedding_len, n_tasks, target_shape, device):
        super(FiLMBlock, self).__init__()

        self._target_shape = target_shape
        self._n_tasks = n_tasks

        self.embedding = nn.Embedding(
            n_tasks, task_embedding_len, device=device)
        self._beta_model = Block(
            task_embedding_len, target_shape, device)
        self._gamma_model = Block(
            task_embedding_len, target_shape, device)

    def forward(self, x: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)
        task_embedding = self.embedding(secret[0, :])
        gamma = self._beta_model(task_embedding)
        beta = self._gamma_model(task_embedding)

        # x_padded = self.pad_to_shape(x)
        x = gamma * x + beta
        # x = self.get_unpadded_back(x_padded, x.shape)
        x = x.permute(1, 0, 2)
        return x

    def pad_to_shape(self, x: torch.Tensor):
        if self._target_shape[1] - x.shape[2] == 0 and self._target_shape[0] - x.shape[1] == 0:
            return x

        return F.pad(x, (0, self._target_shape[1] - x.shape[2], 0, self._target_shape[0] - x.shape[1]), value=0)

    def get_unpadded_back(self, x: torch.Tensor, orig_shape: torch.Size):
        return x[:, :orig_shape[0], :orig_shape[1]]


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(
            torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def generate_square_subsequent_mask(sz, device):
    # Generates a squeare matrix where the each row allows one word more to be seen
    # Lower triangular matrix
    mask = torch.tril(torch.ones(sz, sz) == 1)
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float(
        '-inf'))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

    return mask.to(device)


def create_mask(src, tgt):
    """
    Create attention masks for batched source and target sequences.

    Args:
        src (torch.Tensor): Batched source sequences.
        tgt (torch.Tensor): Batched target sequences.

    Returns:
        src_attention_mask (torch.Tensor): Attention mask for source sequences.
        tgt_attention_mask (torch.Tensor): Attention mask for target sequences.
    """

    subseq_mask = generate_square_subsequent_mask(tgt.shape[1], tgt.device)
    return subseq_mask
