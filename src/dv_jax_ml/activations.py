import jax 
import jax.numpy as jnp


def RELU(x):
    return jnp.maximum(0, x)

def LEAKY_RELU(x, n_slope =.01):
    return jnp.where(x > 0, x, n_slope * x)

def GELU(x): 
    return .5* x * (1 + jax.lax.erf(x/ jnp.sqrt(2.0)))

def SIGMOID(x):
    return 1.0/ (1.0 +jnp.exp(-x)) 

def TANH(x):
    return jnp.tanh(x)

def SOFTMAX(x, axis = -1):
    exp_x = jnp.exp(x - jnp.max(x, axis = axis, keepdims= True) )

    return exp_x/ jnp.sum(exp_x, axis = axis , keepdims = True)
