import jax 
import jax.numpy as jnp

from .normalization import BatchNorm
from .attention import MultiHeadAttention


class Sequential():
    def __init__(self, *layers):
        self.layers = layers

    def init(self, key=jax.random.PRNGKey(42)):
        keys = jax.random.split(key, len(self.layers))
        params = {}
        state = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "init"): 
                res = layer.init(keys[i])
                if isinstance(res, tuple) and len(res) == 2:
                    params[str(i)], state[str(i)] = res
                else:
                    params[str(i)] = res
                    state[str(i)] = {}
            else:
                params[str(i)] = {}
                state[str(i)] = {}
        return params, state

    def apply(self, params, state, x, key=None, training=False, **kwargs): 
        new_state = {}
        for i, layer in enumerate(self.layers):
            layer_key = None
            if key is not None:
                key, layer_key = jax.random.split(key)

            l_params = params.get(str(i), {})
            l_state = state.get(str(i), {})

            if hasattr(layer, 'apply'):
                if isinstance(layer, (BatchNorm, Sequential, ResBlock, MultiHeadAttention)):
                    x, updated_s = layer.apply(
                        l_params, 
                        l_state,
                        x,
                        key=layer_key,
                        training=training, 
                        **kwargs
                    )
                    new_state[str(i)] = updated_s
                else:
                    x = layer.apply(
                        l_params, 
                        x,
                        key=layer_key,
                        training=training ,
                        **kwargs
                    )
                    new_state[str(i)] = l_state
            else: 
                x = layer(x)
                new_state[str(i)] = l_state
        return x, new_state

    __call__ = apply


class ResBlock:
    def __init__(self, layer, shortcut=None):
        self.layer = layer 
        self.shortcut = shortcut

    def init(self, key=jax.random.PRNGKey(42)):
        key_main, key_sc = jax.random.split(key)
        params = {}
        state = {}

        if hasattr(self.layer, 'init'):
            res = self.layer.init(key_main)
            if isinstance(res, tuple) and len(res) == 2:
                params['layer'], state['layer'] = res
            else:
                params['layer'] = res
                state['layer'] = {}
        else:
            params['layer'] = {}
            state['layer'] = {}

        if self.shortcut is not None and hasattr(self.shortcut, 'init'):
            res_sc = self.shortcut.init(key_sc)
            if isinstance(res_sc, tuple) and len(res_sc) == 2:
                params['shortcut'], state['shortcut'] = res_sc
            else:
                params['shortcut'] = res_sc
                state['shortcut'] = {}
        elif self.shortcut is not None:
            params['shortcut'] = {}
            state['shortcut'] = {}
        
        return params, state

    def apply(self, params, state, x, key=None, training=False, **kwargs):
        k1, k2 = (None, None) if key is None else jax.random.split(key)
        new_state = {}

        layer_params = params.get('layer', {})
        layer_state = state.get('layer', {})

        if hasattr(self.layer, 'apply'):
            if isinstance(self.layer, (BatchNorm, Sequential, ResBlock,MultiHeadAttention)):
                out, new_layer_state = self.layer.apply(
                    layer_params, 
                    layer_state,
                    x, 
                    key=k1,
                    training=training,
                    **kwargs
                )
                new_state['layer'] = new_layer_state
            else:
                out = self.layer.apply(
                    layer_params, 
                    x, 
                    key=k1,
                    training=training,
                    **kwargs
                )
                new_state['layer'] = layer_state
        else:
            out = self.layer(x)
            new_state['layer'] = layer_state

        if self.shortcut is not None:
            sc_params = params.get('shortcut', {})
            sc_state = state.get('shortcut', {})
            if hasattr(self.shortcut, 'apply'):
                if isinstance(self.shortcut, (BatchNorm, Sequential, ResBlock, MultiHeadAttention)):
                    res, new_sc_state = self.shortcut.apply(
                        sc_params,
                        sc_state,
                        x, 
                        key=k2, 
                        training=training,
                        **kwargs 
                    )
                    new_state['shortcut'] = new_sc_state
                else:
                    res = self.shortcut.apply(
                        sc_params,
                        x, 
                        key=k2, 
                        training=training,
                        **kwargs 
                    )
                    new_state['shortcut'] = sc_state
            else:
                res = self.shortcut(x)
                new_state['shortcut'] = sc_state
        else:
            res = x
        return out + res, new_state

    __call__ = apply
