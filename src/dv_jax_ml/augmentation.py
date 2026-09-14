import jax 
import jax.numpy as jnp


def augment_batch(batch_x, key):
    #made for images, do not use otherwise. 
    k1, k2, k3 = jax.random.split(key, 3)
    flip_mask = jax.random.bernoulli(k1, p=0.5, shape=(batch_x.shape[0], 1, 1, 1))
    batch_x = jnp.where(flip_mask, batch_x[:, :, ::-1, :], batch_x)
    

    padded = jnp.pad(batch_x, ((0,0), (4,4), (4,4), (0,0)), mode='reflect')
    
    h_start = jax.random.randint(k2, (batch_x.shape[0],), 0, 9)
    w_start = jax.random.randint(k3, (batch_x.shape[0],), 0, 9)
    
    def crop_single(img, h, w):
        return jax.lax.dynamic_slice(img, (h, w, 0), (32, 32, 3))
        
    return jax.vmap(crop_single)(padded, h_start, w_start)
