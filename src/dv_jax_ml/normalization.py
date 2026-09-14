import jax 
import jax.numpy as jnp


class LayerNorm:
    def __init__(self, num_features: int, eps: float = 1e-5):
        self.num_features = num_features
        self.eps = eps
    def init(self, key= jax.random.PRNGKey(42)):
        return {
            'gamma': jnp.ones((self.num_features,)),
            'beta': jnp.zeros((self.num_features,))
        }
    def apply(self, params, x, **kwargs):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return params['gamma'] * (x - mean) / jnp.sqrt(var + self.eps) + params['beta']
    __call__ = apply


class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features 
        self.eps = eps
        self.momentum = momentum

    def init(self, key=None):
        params = {
            'gamma': jnp.ones((self.num_features,)),
            'beta': jnp.zeros((self.num_features,))
        }
        state = {
            'running_mean': jnp.zeros((self.num_features,)),
            'running_var': jnp.ones((self.num_features,))
        }
        return params, state

    def apply(self, params, state, x, training=False, **kwargs):
        reduction_axes = tuple(range(x.ndim - 1))
        momentum = kwargs.get('momentum', self.momentum)

        if training:
            mean = jnp.mean(x, axis=reduction_axes, keepdims=True)
            var = jnp.var(x, axis=reduction_axes, keepdims=True)

            batch_mean = mean.reshape((self.num_features,))
            batch_var = var.reshape((self.num_features,))

            running_mean = (1.0 - momentum) * state['running_mean'] + momentum * batch_mean
            running_var = (1.0 - momentum) * state['running_var'] + momentum * batch_var

            x_norm = (x - mean) / jnp.sqrt(var + self.eps)
            new_state = {
                'running_mean': running_mean,
                'running_var': running_var
            }
        else:
            broadcast_shape = (1,) * (x.ndim - 1) + (self.num_features,)
            r_mean = state['running_mean'].reshape(broadcast_shape)
            r_var = state['running_var'].reshape(broadcast_shape)

            x_norm = (x - r_mean) / jnp.sqrt(r_var + self.eps)
            new_state = state

        out = params['gamma'] * x_norm + params['beta']
        return out, new_state

    __call__ = apply
