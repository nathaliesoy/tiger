import math
import torch
import torch.nn as nn

import torch.nn as nn
from torch.nn.functional import softmax, pad
import numpy as np



def add_dims(x, ndim):
    """Adds dimensions to a tensor to match the shape of another tensor."""
    if (dim_diff := ndim - x.dim()) < 0:
        raise ValueError(f"Target ndim ({ndim}) is larger than input ndim ({x.dim()})")

    if dim_diff > 0:
        x = x.view(x.shape[0], *dim_diff * (1,), *x.shape[1:])

    return x



def masked_softmax(x, mask, dim = -1):
    """Applies softmax over a tensor without including padded elements."""
    if mask is not None:
        mask = add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)

    x = softmax(x, dim=dim)

    if mask is not None:
        x = x.masked_fill(mask, 0)

    return x



def merge_masks(
    q_mask,
    kv_mask,
    attn_mask,
    q_shape,
    k_shape,
    device,
):
    """Create a full attention mask which incoporates the padding information.

    Using pytorch transformer convention:
        False: Real node
        True:  Zero padded
    """
    # Create the full mask which combines the attention and padding masks
    merged_mask = None

    # If either pad mask exists, create
    if q_mask is not None or kv_mask is not None:
        if q_mask is None:
            q_mask = torch.full(q_shape[:-1], False, device=device)
        if kv_mask is None:
            kv_mask = torch.full(k_shape[:-1], False, device=device)
        merged_mask = q_mask.unsqueeze(-1) | kv_mask.unsqueeze(-2)

    # If attention mask exists then it must be included
    if attn_mask is not None:
        merged_mask = attn_mask if merged_mask is None else attn_mask | merged_mask

    return merged_mask




def get_random_mask(n_batch: int, n_track: int, p_valid: float = 0.5):
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    a = rng.choice(a=[True, False], size=(n_batch, n_track), p=[1 - p_valid, p_valid])
    a = np.sort(a, axis=-1)[:, ::-1]
    if n_track > 0 and p_valid > 0:  # ensure at least one valid track
        a[:, 0] = False
    return torch.tensor(a.copy())



def attach_context(x, context):
    """Concatenates a context tensor to an input tensor with considerations for
    broadcasting.

    The idea behind this is to allow a context tensor less or equal dimensions to be
    concatenated to an input with more dimensions.
    This function checks the dimension difference and reshapes the context to have
    the same dimension as the constituents so broadcast concatenation can apply.
    The shape change always assumes that the first dimension is the batch and the last
    are the features.

    Here is a basic use case: concatenating the high level jet variables to the
    constituents during a forward pass through the network
    - The jet variables will be of shape: [batch, j_features]
    - The constituents will be of shape: [batch, num_nodes, n_features]
    Here the jet variable will be shaped to [batch, 1, j_features] allowing broadcast
    concatenation with the constituents

    Another example is using the edge features [b,n,n,ef] and concatenating the
    high level jet variables, which will be expanded to [b,1,1,jf] or the conditioning
    on the node features which will be expanded to [b,1,n,nf]
    """
    if context is None:
        raise RuntimeError("Expected context is missing from forward pass")

    # Check if the context information has less dimensions and the broadcast is needed
    if (dim_diff := x.dim() - context.dim()) < 0:
        raise ValueError(
            f"Provided context has more dimensions ({context.dim()}) than inputs ({x.dim()})"
        )

    # If reshaping is required
    if dim_diff > 0:
        # Reshape the context inputs with 1's after the batch dim
        context = add_dims(context, x.dim())

        # Use expand to allow for broadcasting as expand does not allocate memory
        context = context.expand(*x.shape[:-1], -1)

    # Apply the concatenation on the final dimension
    return torch.cat([x, context], dim=-1)



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def padded_to_packed(seq, inv_mask):
    # inv_mask: True for valid tokens
    seqlens = inv_mask.sum(dim=-1)
    maxlen = seqlens.max() # .item()
    culens = pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return seq[inv_mask], culens, maxlen


def packed_to_padded(unpadded_seq, inv_mask):
    # inv_mask: True for valid tokens
    shape = (*inv_mask.shape, unpadded_seq.shape[-1])
    out = torch.zeros(shape, dtype=unpadded_seq.dtype, device=unpadded_seq.device)
    out[inv_mask] = unpadded_seq
    return out



def partitionfunc(n,k,l=1):
    '''
    n is the integer to partition, k is the length of partitions, l is the min partition element size
    Adapted from https://stackoverflow.com/questions/18503096/python-integer-partitioning-with-given-k-partitions
    '''
    if k < 1:
        return
    if k == 1:
        if n >= l:
            yield (n,)
        return
    for i in range(l,n+1):
        for result in partitionfunc(n-i,k-1,i):
            yield (i,)+result

class IntegerPartitionSampler:
    def __init__(self, n, k, rng):
        self.partitions = np.array(list(partitionfunc(n, k, 0)))
        self.rng = rng

    def __call__(self):
        return self.rng.permutation(self.rng.choice(self.partitions))