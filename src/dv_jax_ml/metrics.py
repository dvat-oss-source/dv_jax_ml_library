import jax 
import jax.numpy as jnp

from .losses import cross_entropy_loss
from .training import evaluate


def binary_roc_auc(true, score):
    desc_indices = jnp.argsort(-score)
    true_sorted = true[desc_indices]

    tps = jnp.cumsum(true_sorted)
    fps = jnp.cumsum(1.0-true_sorted)

    total_pos = tps[-1]
    total_neg = fps[-1]

    tpr = jnp.concatenate([jnp.array([0.0]), tps / jnp.maximum(total_pos, 1e-8)])
    fpr = jnp.concatenate([jnp.array([0.0]), fps / jnp.maximum(total_neg, 1e-8)])
    
    dx = fpr[1:] - fpr[:-1]
    avg_y = (tpr[1:] + tpr[:-1]) / 2.0
    auc = jnp.sum(dx * avg_y)
    
    return auc, fpr, tpr

    

def multiclass_auc_roc(true_onehot, probs):
    
    class_aucs, fprs, tprs = jax.vmap(binary_roc_auc, in_axes=(1, 1))(true_onehot, probs)
    macro_auc = jnp.mean(class_aucs)
    
    return macro_auc #, class_aucs, fprs, tprs


def percent_accuracy(target, pred):
    if target.ndim>1 :
        target = jnp.argmax(target, axis=-1)
    if pred.ndim>1 :
        pred = jnp.argmax(pred, axis=-1)
    return jnp.mean(target == pred )*100

def perplexity(model, params, state, x , y, loss_func = cross_entropy_loss(), batch_size= 128):
    val_loss = evaluate(
        model, 
        params, 
        state, 
        x, 
        y, 
        loss_func, 
        batch_size=batch_size
    )
    perplexity_score = jnp.exp(val_loss)
    print(f"Test Loss: {val_loss:.4f}")
    print(f"Perplexity: {perplexity_score:.2f}")
    return perplexity_score
