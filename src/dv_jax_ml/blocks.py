from .layers import Linear, Dropout_Layer
from .activations import GELU
from .attention import MultiHeadAttention, MultiHeadLatentAttention
from .normalization import LayerNorm
from .containers import Sequential, ResBlock


def transformer_block(d_model: int, num_heads: int, mlp_dim: int, max_seq_len: int = 0, dropout_rate: float = 0.1, causal: bool = True):
    return Sequential(
        ResBlock(
            Sequential(
                LayerNorm(d_model),
                MultiHeadAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, dropout=dropout_rate, causal=causal),

            )
        ),
        ResBlock(
            Sequential(
                LayerNorm(d_model),
                Linear(d_model, mlp_dim),
                GELU,
                Dropout_Layer(dropout_rate),
                Linear(mlp_dim, d_model),
                Dropout_Layer(dropout_rate),

                
                

            )
        )
    )
        


def transformer_latent_block(d_model: int, num_heads: int,kv_latent_dim: int, mlp_dim: int, q_latent_dim: int = None, max_seq_len: int = 0,dropout_rate: float = 0.1):
    return Sequential(
        ResBlock(
            Sequential(
                LayerNorm(d_model),
                MultiHeadLatentAttention(d_model=d_model, num_heads=num_heads, kv_latent_dim = kv_latent_dim, q_latent_dim = q_latent_dim ,max_seq_len=max_seq_len, dropout=dropout_rate),

            )
        ),
        ResBlock(
            Sequential(
                LayerNorm(d_model),
                Linear(d_model, mlp_dim),
                GELU,
                Dropout_Layer(dropout_rate),
                Linear(mlp_dim, d_model),
                Dropout_Layer(dropout_rate),

                
                

            )
        )
    )
