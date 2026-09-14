import jax 
import jax.numpy as jnp

from .losses import cross_entropy_loss, compute_loss
from .optimizers import SGD


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.best_params = None
        self.best_state = None
        self.counter = 0
        self.should_stop = False
        
    def check(self, val_loss, params, state=None):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.best_params = params
            self.best_state = state
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def make_eval_step(model, loss_func):
    @jax.jit
    def eval_step(params, state, bx, by):
        preds, _ = model.apply(params, state, bx, training=False)
        return loss_func(preds, by)
    return eval_step
def evaluate(model, params, state, x, y, loss_func, batch_size=256, eval_step= None):
    eval_step = make_eval_step(model, loss_func)
    num_samples = x.shape[0]
    num_batches = int(jnp.ceil(num_samples / batch_size))
    total_loss = 0.0

    for i in range(num_batches):
        bx = x[i * batch_size : (i + 1) * batch_size]
        by = y[i * batch_size : (i + 1) * batch_size]

        batch_loss = eval_step(params, state, bx, by)
        total_loss += float(batch_loss) * len(bx)
    return total_loss / num_samples


def train(model, 
    params, 
    x, 
    y, 
    state = None,
    val_data = None,
    patience = 15, 
    min_delta = 1e-4,
    optimizer = SGD(), 
    epochs = 67, 
    batch_size = 128, 
    loss_func = cross_entropy_loss(), 
    key = jax.random.PRNGKey(42),
    augment_fn = None
    
    ):

    if state is None:
        state = {}
    
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta) if val_data is not None else None
    opt_state = optimizer.init(params) if hasattr(optimizer, "init") else None

    @jax.jit
    def step(params, state, opt_state, x, y, key):
        step_key, next_key = jax.random.split(key)

        if augment_fn is not None:
            x = augment_fn(x, next_key)

        (loss, new_state), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            params, state, model, x, y, loss_func, step_key
        )
        

        new_params, new_opt_state = optimizer.update(params, grads, opt_state)

        
        return new_params, new_state, new_opt_state, loss, next_key
    @jax.jit
    def eval_step(params, state, bx, by):
        preds, _ = model.apply(params, state, bx, training=False)
        return loss_func(preds, by)


    num_samples = x.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        key, perm_key = jax.random.split(key)
        perms = jax.random.permutation(perm_key, num_samples)
        epoch_loss = 0.0

        for i in range(num_batches):
            batch_idx = perms[i * batch_size : (i + 1) * batch_size]
            batch_x = x[batch_idx]
            batch_y = y[batch_idx]
            
            params, state, opt_state, loss, key = step(params, state, opt_state, batch_x, batch_y, key)
            epoch_loss += loss 
        
        avg_loss = epoch_loss / num_batches

        if val_data is not None: 
            val_x, val_y = val_data
            val_loss = evaluate(
                model,
                params, 
                state,
                val_x,
                val_y,
                loss_func,
                batch_size = batch_size,
                eval_step = eval_step 
            )

            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f} Val Loss = {val_loss:.4f}")

            if early_stopping.check(val_loss, params, state): 
                print(f'Early stop triggered at epoch {epoch} restoring {early_stopping.best_loss}')
                if early_stopping.restore_best_weights and early_stopping.best_params is not None:
                    return early_stopping.best_params, early_stopping.best_state
                return params, state   
        else:
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

    if early_stopping and early_stopping.restore_best_weights and early_stopping.best_params is not None:
        return early_stopping.best_params, early_stopping.best_state
    return params, state
