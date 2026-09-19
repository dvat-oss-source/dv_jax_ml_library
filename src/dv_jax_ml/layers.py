import jax 
import jax.numpy as jnp


class Linear(): 
    def __init__(self, input, output, key = jax.random.PRNGKey(42)):

        self.input = input
        self.output = output

    def init(self, key = jax.random.PRNGKey(42)):
        key_w, key_b = jax.random.split(key)
        std = jnp.sqrt(2.0 / self.input)


        return {
            'weights': jax.random.normal(key_w, (self.input, self.output)) * std,
            'bias': jnp.zeros((self.output,))
        }

    def __call__(self, params, x, **kwargs):
        return x @params['weights'] +params['bias']

    apply = __call__ 


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size = (3,3), stride = (1,1), padding = "same", groups = 1):
        assert in_channels % groups == 0, f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        assert out_channels % groups == 0, f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance( kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance( stride, tuple) else (stride, stride)
        self.padding = padding
        self.groups = groups

    def init(self, key=jax.random.PRNGKey(42) ):
        key_w, key_b = jax.random.split(key)

        fan_in = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels //self.groups)

        std = jnp.sqrt(2.0/ fan_in)

        kernel_shape = (*self.kernel_size, self.in_channels//self.groups, self.out_channels)

        return {
            'weights' : jax.random.normal(key_w, kernel_shape) * std, 
            'bias' : jnp.zeros((self.out_channels, ))
        }
    def apply(self, params, x, **kwargs):
        out = jax.lax.conv_general_dilated(
            lhs = x,
            rhs = params['weights'],
            window_strides = self.stride, 
            padding = self.padding,
            dimension_numbers=  ('NHWC', 'HWIO', 'NHWC' ),
            feature_group_count = self.groups

        )
        return out + params['bias']
    __call__ = apply 
class ConvTranspose2D:
    def __init__(self, in_channels, out_channels, kernel_size = (3,3), stride = (1,1), padding = "same", groups = 1):
       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance( kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance( stride, tuple) else (stride, stride)
        self.padding = padding.upper()
        self.groups = groups

    def init(self, key=jax.random.PRNGKey(42) ):
        key_w, key_b = jax.random.split(key)

        fan_in = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels //self.groups)

        std = jnp.sqrt(2.0/ fan_in)

        kernel_shape = (*self.kernel_size, self.out_channels // self.groups, self.in_channels)

        return {
            'weights' : jax.random.normal(key_w, kernel_shape) * std, 
            'bias' : jnp.zeros((self.out_channels, ))
        }
    def apply(self, params, x, **kwargs ):
        out = jax.lax.conv_transpose(
            lhs=x,
            rhs=params['weights'],
            strides=self.stride,
            padding=self.padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            transpose_kernel=True,
        )
        return out + params['bias'] 
    __call__ = apply 
        

        

class Dropout_Layer:
    def __init__(self, rate =.5):
        self.rate = rate

    def apply(self, params, x, key = None, training = False, **kwargs):
        if not training or self.rate  <= 0.0 or key is None:
            return x
        keep_prob = 1.0- self.rate 
        mask = jax.random.bernoulli(key, p = keep_prob, shape = x.shape)
        return (x*mask) / keep_prob
    __call__ = apply


class Flatten:
    def apply(self, params, x, **kwargs):
        return x.reshape(x.shape[0], -1 )
    __call__ = apply


class MaxPool2D:
    def __init__(self, window_shape = (2,2) , strides= (2,2), padding ="VALID" ): 

        self.window_shape = window_shape if isinstance(window_shape, tuple) else (window_shape, window_shape)
        self.strides =  strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding
    def apply(self, params, x, **kwargs): 
        return jax.lax.reduce_window(
            x,
            -jnp.inf,
            jax.lax.max,
            window_dimensions = (1, *self.window_shape, 1),
            window_strides = (1, *self.strides, 1), 
            padding = self.padding
        )
    __call__ = apply 
 

class GlobalAvgPool2D:
    def apply(self, params, x,**kwargs):
        return jnp.mean(x, axis=(1, 2))

    __call__ = apply


    

