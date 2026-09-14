import jax 
import jax.numpy as jnp


class SGD:
    def  __init__(self,  lr = .05):
        self.lr = lr
    
    def init(self, params):
        state = {
            'step': jnp.array(0, dtype = jnp.int32)
        }
        return state
        
    def update(self, params, grads, state):
        t = state['step'] + 1
        current_lr = self.lr(t) if callable(self.lr) else self.lr
        new_params = jax.tree.map(
            lambda p, g: p - current_lr * g, params, grads
        )
        return new_params, {'step': t}


class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay

    def init(self, params):
        state = {
            'step': 0,
            'm': jax.tree.map(lambda p: jnp.zeros_like(p), params),
            'v': jax.tree.map(lambda p: jnp.zeros_like(p), params),
        }
        return state

    def update(self, params, grads, state):
        t = state['step'] + 1 

        current_lr = self.lr(t) if callable(self.lr) else self.lr

        m_new = jax.tree.map(
            lambda m, g: self.b1 * m + (1 - self.b1) * g, 
            state['m'], 
            grads
        )
        v_new = jax.tree.map(
            lambda v, g: self.b2 * v + (1 - self.b2) * (g**2), 
            state['v'], 
            grads
        ) 

        m_hat = jax.tree.map(
            lambda m: m / (1.0 - self.b1 ** t),
            m_new
        )
        v_hat = jax.tree.map(
            lambda v: v / (1.0 - self.b2 ** t), 
            v_new
        )

        def update_param(p, m_h, v_h):
            step_update = current_lr * m_h / (jnp.sqrt(v_h) + self.eps)
            if self.weight_decay > 0.0 and p.ndim >= 2:
                return p - current_lr * self.weight_decay * p - step_update
            return p - step_update
        params_new = jax.tree.map(update_param, params, m_hat, v_hat)
      

        new_state = {
            'step': t, 
            'm': m_new,
            'v': v_new,
        }
        return params_new, new_state
