import jax.numpy as jnp


#LR schedulers, hopefully in a future implemnetation I can add a drop on plateau. However I've generally had a lot of trouble adding it to the stateless architecure of this library and use of jit compiling. I might need to fully redo the training loop.
def cosine_decay(init_lr, max_lr, min_lr, warmup_steps, total_steps):
    def schedule(step):
        warmup_lr = init_lr + (max_lr - init_lr) * step / jnp.maximum(1, warmup_steps)
        
        decay_steps = jnp.maximum(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        progress = jnp.clip(progress, 0.0, 1.0)
        cosine_lr = min_lr +.5 * (max_lr - min_lr) * (1.0 +jnp.cos(jnp.pi * progress))

        return jnp.where(step< warmup_steps, warmup_lr, cosine_lr)

        
    return schedule 
