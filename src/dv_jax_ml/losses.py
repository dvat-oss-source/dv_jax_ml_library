import jax 
import jax.numpy as jnp


class MSEloss: 
    def __call__(self, predictions, y):
         return jnp.mean((predictions- y)**2)



class cross_entropy_loss:
    def __call__ (self, predictions, y, eps=1e-7):
        log_probs = jax.nn.log_softmax(predictions, axis=-1)

        if y.ndim == predictions.ndim - 1:
            target_log_probs = jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
            return -jnp.mean(target_log_probs)
        return -jnp.mean(jnp.sum(y * log_probs, axis=-1))

def compute_loss(params, state, model, x, y, loss_func, key=None):
    predictions, new_state = model.apply(params, state, x, key=key, training=True)
    return loss_func(predictions, y), new_state
