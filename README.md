# dv-jax-ml

A from-scratch deep learning library built on [JAX](https://github.com/jax-ml/jax) with JIT compilation. No high-level frameworks 

## Features

- **Layers**: `Linear`, `Conv2D`, `Flatten`, `MaxPool2D`, `GlobalAvgPool2D`, `Dropout_Layer`
- **Activations**: `RELU`, `LEAKY_RELU`, `GELU`, `SIGMOID`, `TANH`, `SOFTMAX`
- **Attention**: `MultiHeadAttention` (with KV cache), `MultiHeadLatentAttention` (DeepSeek style )
- **Normalization**: `LayerNorm`, `BatchNorm`
- **Containers**: `Sequential`, `ResBlock`
- **Embeddings**: `PatchEmbedding` (ViT-style), `TextEmbedding` (token + positional)
- **Blocks**: `transformer_block`, `transformer_latent_block` (pre-norm transformer blocks)
- **Optimizers**: `SGD`, `Adam` (with weight decay)
- **Schedulers**: `cosine_decay` (with warmup)
- **Training**: `train` (full training loop and early stopping), `evaluate`
- **Inference**: `predict`, `predict_proba`
- **Metrics**: `binary_roc_auc`, `multiclass_auc_roc`, `percent_accuracy`, `perplexity`
- **Augmentation**: `augment_batch` (random flip + crop for images)

## Installation

# Installing from GitHub
pip install git+https://github.com/dvat-oss-source/dv_jax_ml_library.git

## Architecture

The library uses a stateless design pattern:
- **Params** are plain pytree dicts (JAX-friendly for `jit`, `grad`, `vmap`)
- **State** is a separate dict for stateful layers (e.g., `BatchNorm` running stats, KV caches)
- Every layer exposes `.init(key)`  params and `.apply(params, x)` output

```
src/dv_jax_ml/
├── activations.py      # Activation functions
├── attention.py         # Multi-head attention (standard + latent)
├── augmentation.py      # Data augmentation
├── blocks.py            # Pre-built transformer blocks
├── containers.py        # Sequential, ResBlock
├── embeddings.py        # Patch and text embeddings
├── inference.py         # predict, predict_proba
├── layers.py            # Linear, Conv2D, pooling, dropout
├── losses.py            # MSE, cross-entropy
├── metrics.py           # AUC-ROC, accuracy, perplexity
├── normalization.py     # LayerNorm, BatchNorm
├── optimizers.py        # SGD, Adam
├── schedulers.py        # Cosine decay with warmup
└── training.py          # Training loop, evaluation, early stopping
```

## License

MIT

#Demo code for Mnist and WikiText-2 perferably run this in a notebook. 

```python
# %%
from dv_jax_ml import Sequential, Linear, Conv2D, BatchNorm, Adam, train, cross_entropy_loss, GELU, MaxPool2D, ResBlock, GlobalAvgPool2D, cosine_decay
import jax 
import jax.numpy as jnp



# %%
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras


# %%
from keras.datasets import mnist
data = mnist.load_data()

(X_train, y_train), (X_test, y_test) = data
 

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')

X_test = X_test.reshape((X_test.shape[0],  28, 28, 1)).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = jax.nn.one_hot(y_train, 10)
y_test = jax.nn.one_hot(y_test, 10)


# %%
basic_model = Sequential(
    
    Conv2D(in_channels = 1, out_channels = 128 , kernel_size = (7, 7 ),  stride = (2 ,2), padding ='same'),
    BatchNorm(128), 
    GELU, 

    MaxPool2D(window_shape = (7,7), strides = (2,2)),
    
    ResBlock(
        Sequential(
            Conv2D(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding="same"),
            BatchNorm(64),
            GELU,

            Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="same", groups=16),
            BatchNorm(64),
            GELU,

            Conv2D(in_channels = 64, out_channels = 128 , kernel_size = (1,1) ,stride = (1,1),  padding ='same'), 
            BatchNorm(128)         
        ),


        shortcut=Conv2D(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding="same")
    ),
    GELU,

    GlobalAvgPool2D(),      
    Linear(input=128, output=10),
    #SOFTMAX
    
)


# %%
steps_per_epoch = len(X_train) // 128
epochs = 3
total_steps = steps_per_epoch * epochs
warmup_steps = total_steps // 10  

key = jax.random.PRNGKey(42)

params, state = basic_model.init(key)

params, state = train(basic_model,
   params, 
   X_train, 
   y_train, 
   state = state,
   val_data = (X_test, y_test),
   patience = 15, 
   min_delta = 1e-4,
   optimizer = Adam(
      lr = cosine_decay(
      init_lr= 0.0001,
      max_lr= 0.001,
      min_lr= 0.000001,
      warmup_steps= warmup_steps, 
      total_steps=  total_steps
   )
   ),
   epochs = epochs, 
   batch_size = 128,
   loss_func=cross_entropy_loss()
   )


# %%

from dv_jax_ml import predict_proba, predict, multiclass_auc_roc, percent_accuracy, perplexity

probs = predict_proba(basic_model, params, state, X_test)

pred = predict(basic_model, params, state, X_test)

print(multiclass_auc_roc(y_test, probs))

percent_accuracy(y_test, pred)



# %%
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

import numpy as np
raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = [text.strip() for text in raw_dataset["train"]["text"] if len(text.strip()) > 10]

vocab_size = 4096
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"], vocab_size=vocab_size)
tokenizer.train_from_iterator(train_texts, trainer=trainer)
all_token_ids = []
for text in train_texts:
    all_token_ids.extend(tokenizer.encode(text).ids)
all_token_ids = np.array(all_token_ids, dtype=np.int32)
print(f"Total training tokens: {len(all_token_ids):,}")

seq_len = 256

num_chunks = len(all_token_ids) // (seq_len + 1)
data = all_token_ids[:num_chunks * (seq_len + 1)].reshape(num_chunks, seq_len + 1)

X_all = data[:, :-1]
y_all = data[:, 1:]

split = int(0.9 * num_chunks)
X_train, y_train = X_all[:split], y_all[:split]
X_test, y_test   = X_all[split:], y_all[split:]


# %%
from dv_jax_ml import TextEmbedding, transformer_latent_block, LayerNorm
vocab_size = 4096


d_model = 256
max_seq_len = 256
num_heads = 4
mlp_dim = 1024

kv_latent_dim = 64   # Compressed KV dimension (saves memory during generation)
q_latent_dim = 64 


basic_model = Sequential(
    TextEmbedding(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len),
    transformer_latent_block(
        d_model=d_model, 
        num_heads=num_heads, 
        kv_latent_dim=kv_latent_dim, 
        q_latent_dim=q_latent_dim,
        mlp_dim=mlp_dim, 
        max_seq_len=max_seq_len, 
        dropout_rate=0.2
    ),
    
    LayerNorm(d_model),
    Linear(d_model, vocab_size)
)


# %%
steps_per_epoch = len(X_train) // 128
epochs = 3
total_steps = steps_per_epoch * epochs
warmup_steps = total_steps // 10  

key = jax.random.PRNGKey(42)

params, state = basic_model.init(key)

params, state = train(basic_model,
   params, 
   X_train, 
   y_train, 
   state = state,
   val_data = (X_test, y_test),
   patience = 15, 
   min_delta = 1e-4,
   optimizer = Adam(
      lr = cosine_decay(
      init_lr= 0.0001,
      max_lr= 0.001,
      min_lr= 0.000001,
      warmup_steps= warmup_steps, 
      total_steps=  total_steps
   )
   ),
   epochs = epochs, 
   batch_size = 128,
   loss_func=cross_entropy_loss()
   )


# %%


perplexity(
    basic_model, 
    params, 
    state, 
    X_test, 
    y_test, 
)



```
