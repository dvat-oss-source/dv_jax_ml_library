# dv_jax_ml — A JAX-based deep learning library

from .activations import RELU, LEAKY_RELU, GELU, SIGMOID, TANH, SOFTMAX
from .layers import Linear, Conv2D, Dropout_Layer, Flatten, MaxPool2D, GlobalAvgPool2D
from .normalization import LayerNorm, BatchNorm
from .attention import MultiHeadAttention, MultiHeadLatentAttention
from .containers import Sequential, ResBlock
from .embeddings import PatchEmbedding, TextEmbedding
from .blocks import transformer_block, transformer_latent_block
from .losses import MSEloss, cross_entropy_loss, compute_loss
from .optimizers import SGD, Adam
from .schedulers import cosine_decay
from .training import train, evaluate, make_eval_step, EarlyStopping
from .inference import predict, predict_proba
from .metrics import binary_roc_auc, multiclass_auc_roc, percent_accuracy, perplexity
from .augmentation import augment_batch

__version__ = "0.1.0"
