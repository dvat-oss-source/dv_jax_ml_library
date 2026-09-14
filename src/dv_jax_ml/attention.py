import jax 
import jax.numpy as jnp


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 0, dropout: float = 0.0, bias: bool = True):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.headdim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.bias = bias

    def init(self, key=jax.random.PRNGKey(42), batch_size=1):
        keys = jax.random.split(key, 4)
        std = jnp.sqrt(2.0 / self.d_model)
        
        params = {
            'w_q': jax.random.normal(keys[0], (self.d_model, self.d_model)) * std,
            'w_k': jax.random.normal(keys[1], (self.d_model, self.d_model)) * std,
            'w_v': jax.random.normal(keys[2], (self.d_model, self.d_model)) * std,
            'w_o': jax.random.normal(keys[3], (self.d_model, self.d_model)) * std,
        }
        if self.bias:
            params['b_q'] = jnp.zeros((self.d_model,))
            params['b_k'] = jnp.zeros((self.d_model,))
            params['b_v'] = jnp.zeros((self.d_model,))
            params['b_o'] = jnp.zeros((self.d_model,))

        state = {}
        if self.max_seq_len > 0:
            state['k_cache'] = jnp.zeros((batch_size, self.num_heads, self.max_seq_len, self.headdim))
            state['v_cache'] = jnp.zeros((batch_size, self.num_heads, self.max_seq_len, self.headdim))
            state['index'] = jnp.array(0, dtype=jnp.int32)

        return params, state

    def apply(self, params, state, x, mask=None, use_cache=False, key=None, training=False, **kwargs):
        q_proj = x @ params['w_q']
        k_proj = x @ params['w_k']
        v_proj = x @ params['w_v']

        if self.bias:
            q_proj += params['b_q']
            k_proj += params['b_k']
            v_proj += params['b_v']
            
        batch_size, seq_len_q = q_proj.shape[:2]
        seq_len_k = k_proj.shape[1]

        q = q_proj.reshape(batch_size, seq_len_q, self.num_heads, self.headdim).swapaxes(1, 2)
        k = k_proj.reshape(batch_size, seq_len_k, self.num_heads, self.headdim).swapaxes(1, 2)
        v = v_proj.reshape(batch_size, seq_len_k, self.num_heads, self.headdim).swapaxes(1, 2)

        new_state = dict(state)

        if use_cache and not training and 'k_cache' in state:
            idx = state['index']
            k_up = jax.lax.dynamic_update_slice(state['k_cache'], k, (0, 0, idx, 0))
            v_up = jax.lax.dynamic_update_slice(state['v_cache'], v, (0, 0, idx, 0))
            new_state['k_cache'] = k_up
            new_state['v_cache'] = v_up
            new_state['index'] = idx + seq_len_k
            k = k_up
            v = v_up

            q_idx = idx + jnp.arange(seq_len_q)[:, None]
            k_idx = jnp.arange(self.max_seq_len)[None, :]
            mask = ((k_idx <= q_idx) & (k_idx < idx + seq_len_k))[None, None, :, :]

        scale = 1.0 / jnp.sqrt(self.headdim)
        scores = jnp.matmul(q, k.swapaxes(-1, -2)) * scale
        if mask is None and not use_cache:
            q_idx = jnp.arange(seq_len_q)[:, None]
            k_idx = jnp.arange(seq_len_k)[None, :]
            mask = (k_idx <= q_idx)[None, None, :, :]
        
        if mask is not None:
            if mask.dtype == jnp.bool_:
                scores = jnp.where(mask, scores, -1e9)
            else:
                scores = scores + mask

        weights = jax.nn.softmax(scores, axis=-1)

        if training and self.dropout > 0.0 and key is not None:
            keep_prob = 1.0 - self.dropout
            drop_mask = jax.random.bernoulli(key, p=keep_prob, shape=weights.shape)
            weights = (weights * drop_mask) / keep_prob

        out = jnp.matmul(weights, v).swapaxes(1, 2).reshape(batch_size, seq_len_q, self.d_model)
        out = out @ params['w_o']
        if self.bias:
            out = out + params['b_o']

        return out, new_state

    __call__ = apply


class MultiHeadLatentAttention(MultiHeadAttention):
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        kv_latent_dim: int, 
        q_latent_dim: int = None, 
        max_seq_len: int = 0, 
        dropout: float = 0.0, 
        bias: bool = True, 
        use_latent_norm: bool = True, 
        eps: float = 1e-5
    ):
        super().__init__(d_model = d_model, num_heads = num_heads, max_seq_len = max_seq_len, dropout = dropout, bias = bias)
        self.kv_latent_dim = kv_latent_dim
        self.q_latent_dim = q_latent_dim if q_latent_dim is not None else kv_latent_dim
        self.use_latent_norm = use_latent_norm
        self.eps = eps

    def init(self, key=jax.random.PRNGKey(42), batch_size=1):
        keys = jax.random.split(key, 6)
        std = lambda din, dout: jnp.sqrt(2.0 / (din + dout))
        total_dim = self.num_heads * self.headdim

        params = {
            'w_dq':  jax.random.normal(keys[0], (self.d_model, self.q_latent_dim)) * std(self.d_model, self.q_latent_dim),
            'w_uq':  jax.random.normal(keys[1], (self.q_latent_dim, total_dim)) * std(self.q_latent_dim, total_dim),

            'w_dkv': jax.random.normal(keys[2], (self.d_model, self.kv_latent_dim)) * std(self.d_model, self.kv_latent_dim),
            'w_uk':  jax.random.normal(keys[3], (self.kv_latent_dim, total_dim)) * std(self.kv_latent_dim, total_dim),
            'w_uv':  jax.random.normal(keys[4], (self.kv_latent_dim, total_dim)) * std(self.kv_latent_dim, total_dim),

            'w_o':   jax.random.normal(keys[5], (total_dim, self.d_model)) * std(total_dim, self.d_model),
        }

        if self.bias:
            params['b_dq']  = jnp.zeros((self.q_latent_dim,))
            params['b_uq']  = jnp.zeros((total_dim,))
            params['b_dkv'] = jnp.zeros((self.kv_latent_dim,))
            params['b_uk']  = jnp.zeros((total_dim,))
            params['b_uv']  = jnp.zeros((total_dim,))
            params['b_o']   = jnp.zeros((self.d_model,))

        if self.use_latent_norm:
            params['gamma_q']  = jnp.ones((self.q_latent_dim,))
            params['gamma_kv'] = jnp.ones((self.kv_latent_dim,))

        state = {}
        if self.max_seq_len > 0:
            state['c_kv'] = jnp.zeros((batch_size, self.max_seq_len, self.kv_latent_dim))
            state['index'] = jnp.array(0, dtype=jnp.int32)

        return params, state

    def _rms_norm(self, x, gamma):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(variance + self.eps) * gamma

    def apply(self, params, state, x, mask=None, use_cache=False, key=None, training=False, **kwargs):
        batch_size, seq_len_q = x.shape[:2]
        total_dim = self.num_heads * self.headdim

        q_latent = x @ params['w_dq']
        if self.bias:
            q_latent = q_latent + params['b_dq']
        if self.use_latent_norm:
            q_latent = self._rms_norm(q_latent, params['gamma_q'])

        q = q_latent @ params['w_uq']
        if self.bias:
            q = q + params['b_uq']
        q = q.reshape(batch_size, seq_len_q, self.num_heads, self.headdim).swapaxes(1, 2)

        k_input = kwargs.get('k', x)
        cur_kv_latent = k_input @ params['w_dkv']
        if self.bias:
            cur_kv_latent = cur_kv_latent + params['b_dkv']
        if self.use_latent_norm:
            cur_kv_latent = self._rms_norm(cur_kv_latent, params['gamma_kv'])

        new_state = dict(state)

        if use_cache and not training and 'c_kv' in state:
            idx = state['index']
            c_kv_up = jax.lax.dynamic_update_slice(state['c_kv'], cur_kv_latent, (0, idx, 0))
            new_state['c_kv'] = c_kv_up
            new_state['index'] = idx + seq_len_q
            kv_all = c_kv_up

            q_idx = idx + jnp.arange(seq_len_q)[:, None]
            k_idx = jnp.arange(self.max_seq_len)[None, :]
            mask = ((k_idx <= q_idx) & (k_idx < idx + seq_len_q))[None, None, :, :]
        else:
            kv_all = cur_kv_latent

        seq_len_kv = kv_all.shape[1]

        k_proj = kv_all @ params['w_uk']
        v_proj = kv_all @ params['w_uv']
        if self.bias:
            k_proj = k_proj + params['b_uk']
            v_proj = v_proj + params['b_uv']

        k = k_proj.reshape(batch_size, seq_len_kv, self.num_heads, self.headdim).swapaxes(1, 2)
        v = v_proj.reshape(batch_size, seq_len_kv, self.num_heads, self.headdim).swapaxes(1, 2)

        scale = 1.0 / jnp.sqrt(self.headdim)
        scores = jnp.matmul(q, k.swapaxes(-1, -2)) * scale
        

        if mask is None and not use_cache:
            q_idx = jnp.arange(seq_len_q)[:, None]
            k_idx = jnp.arange(seq_len_kv)[None, :]
            mask = (k_idx <= q_idx)[None, None, :, :]


        if mask is not None:
            if mask.dtype == jnp.bool_:
                scores = jnp.where(mask, scores, -1e9)
            else:
                scores = scores + mask

        weights = jax.nn.softmax(scores, axis=-1)

        if training and self.dropout > 0.0 and key is not None:
            keep_prob = 1.0 - self.dropout
            drop_mask = jax.random.bernoulli(key, p=keep_prob, shape=weights.shape)
            weights = (weights * drop_mask) / keep_prob

        out = jnp.matmul(weights, v).swapaxes(1, 2).reshape(batch_size, seq_len_q, self.d_model)
        out = out @ params['w_o']
        if self.bias:
            out = out + params['b_o']

        return out, new_state

    __call__ = apply
