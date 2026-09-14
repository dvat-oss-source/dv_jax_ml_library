import jax 
import jax.numpy as jnp


class PatchEmbedding: 
    def __init__(self, in_channels=3, patch_size=4, img_size=32, d_model=128):
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) **2
        self.patch_dim = in_channels * patch_size * patch_size

    
    def init(self, key=jax.random.PRNGKey(42)):
        k_proj, k_pos = jax.random.split(key)
        std = jnp.sqrt(2.0 / self.patch_dim)

        return {
            'proj':  jax.random.normal(k_proj, (self.patch_dim, self.d_model)) * std,
            'bias': jnp.zeros((self.d_model,)),
            'pos_embed':jax.random.normal(k_pos, (1, self.num_patches, self.d_model)) * 0.02
        }
    def apply(self, params, x, **kwargs):
        B, H, W, C = x.shape
        p = self.patch_size
        p = self.patch_size
        patches = x.reshape(B, H // p, p, W // p, p, C).swapaxes(2, 3).reshape(B, -1, p * p * C)

        tokens = patches @  params['proj'] + params['bias']

        return tokens + params['pos_embed'] 

    __call__ = apply 

class TextEmbedding:
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 256):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def init(self, key=jax.random.PRNGKey(42)):
        k_tok, k_pos = jax.random.split(key)
        return {
            'token_embed': jax.random.normal(k_tok, (self.vocab_size, self.d_model)) * (1.0 / jnp.sqrt(self.d_model)),
            'pos_embed': jax.random.normal(k_pos, (self.max_seq_len, self.d_model)) * 0.02,
        }

    def apply(self, params, x, start_idx=0, **kwargs):
        tok_vecs = params['token_embed'][x]
        seq_len = x.shape[1]
        pos_vecs = params['pos_embed'][jnp.arange(start_idx, start_idx + seq_len)][None, :, :]
        return tok_vecs + pos_vecs

    __call__ = apply
