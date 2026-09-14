import jax.numpy as jnp

from .activations import SOFTMAX


def predict(model, params, state, input_x, batch_size=128):
    num_batches = int(jnp.ceil(input_x.shape[0] / batch_size))
    all_preds = []
    for i in range(num_batches):
        bx = input_x[i * batch_size : (i + 1) * batch_size]
        preds, _ = model.apply(params, state, bx, training=False)
        preds = jnp.argmax(preds, axis=-1)
        all_preds.append(preds)
    return jnp.concatenate(all_preds, axis=0)

def predict_proba(model, params, state, input_x, batch_size=128):
    num_batches = int(jnp.ceil(input_x.shape[0] / batch_size))
    all_preds = []
    for i in range(num_batches):
        bx = input_x[i * batch_size : (i + 1) * batch_size]
        logits, _ = model.apply(params, state, bx, training=False)
        probs = SOFTMAX(logits, axis=-1)
        all_preds.append(probs)
    return jnp.concatenate(all_preds, axis=0)
