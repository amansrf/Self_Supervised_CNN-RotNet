from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import flax.linen as nn

ModuleDef = Any
dtypedef = Any

class RotNetBlock(nn.Module):
    """ The basic building block for RotNet. Both Features and Classifier submodule can contain several RotNetBlocks.

    Args:
        cnn_channels (int): number of channels for all Convolutional layers (We keep them the same for simplicity)
        norm (ModuleDef): the BatchNorm layer being passed in
        dtype (dtypedef): the type of data the model will be dealing with
        kernel_init (Callable): the type of initialization to use for initializing the model params

    """
    cnn_channels: int
    norm: ModuleDef
    dtype: dtypedef
    kernel_init: Callable

    @nn.compact
    def __call__(self, x):
        """
            TODO: Implement forward Pass for RotNetBlock.
            Each block should contains:
                1. a convolutional layer
                2. a batch norm layer. hint: use the norm function defined above
                3. a relu layer
            hint: Check out how to define a module using nn.compact here: https://flax.readthedocs.io/en/latest/guides/setup_or_nncompact.html
        """
       ############################## Your Code Starts Here #########################################
        
    
    
       ############################## Your Code Ends Here #########################################
    
class Features(nn.Module):
    """ Features module of a RotNet. A Features module consist of self.num_blocks of RotNetBlock

    Args:
        cnn_channels (int): number of channels for all Convolutional layers (We keep them the same for simplicity)
        num_blocks (int): number of RotNetBlocks to include in Features submodule
        dtype (dtypedef): the type of data the model will be dealing with
        kernel_init (Callable): the type of initialization to use for initializing the model params
        
    """
    cnn_channels: int
    num_blocks: int
    dtype: dtypedef
    kernel_init: Callable
    
    @nn.compact
    def __call__(self, x, train):
        norm = partial(nn.BatchNorm, use_running_average=not train, dtype=self.dtype)
        for _ in range(self.num_blocks):
            x = RotNetBlock(cnn_channels=self.cnn_channels, norm=norm, dtype=self.dtype, kernel_init=self.kernel_init)(x)
        return x

class Classifier(nn.Module):
    """ Features module of a RotNet. A Features module consist of self.num_blocks of RotNetBlock

        Args:
            cnn_channels (int): number of channels for all Convolutional layers (We keep them the same for simplicity)
            num_blocks (int): number of RotNetBlocks to include in Features submodule
            num_classes (int): number of classes (used in the last layer of Classifier submodule)
            dtype (dtypedef): the type of data the model will be dealing with
            kernel_init (Callable): the type of initialization to use for initializing the model params
    """
    cnn_channels: int
    num_blocks: int
    num_classes: int
    dtype: dtypedef
    kernel_init: Callable
    
    @nn.compact
    def __call__(self, x, train):
        norm = partial(nn.BatchNorm, use_running_average=not train, dtype=self.dtype)
        output = x
        # TODO: complete the forward pass by ddding self.num_blocks of RotNetBlocks
        ############################## Your Code Starts Here #########################################
        # hint: Features class should give you an idea how to implement it    
            
        ############################## Your Code Ends Here #########################################
        output = output.reshape(output.shape[0], -1)
        output = nn.Dense(features=self.num_classes, dtype=self.dtype, kernel_init=self.kernel_init)(output)
        return output

class RotNet(nn.Module):
    """ The RotNet class contains two submodules: Features and Classifier.
        The output of Features submodule will be extracted for transfer learning.

    Args:
        cnn_channels (int): number of channels for all Convolutional layers (We keep them the same for simplicity)
        num_blocks_features (int): number of RotNetBlocks to include in Features submodule
        num_blocks_classifier (int): number of RotNetBlocks to include in Classifier submodule
        num_classes (int): number of classes (used in the last layer of Classifier submodule)
        dtype (dtypedef): the type of data the model will be dealing with
        kernel_init (Callable): the type of initialization to use for initializing the model params
    """
    cnn_channels: int
    num_blocks_features: int
    num_blocks_classifier: int
    num_classes: int
    dtype: dtypedef = jnp.float32
    kernel_init: Callable = nn.initializers.glorot_uniform()

    # ---------------------- Refactor Module into Submodules --------------------- #
    # https://flax.readthedocs.io/en/latest/guides/extracting_intermediates.html#refactor-module-into-submodules
    def setup(self):
        self.features = Features(self.cnn_channels, self.num_blocks_features, self.dtype, self.kernel_init)
        self.classifier = Classifier(self.cnn_channels, self.num_blocks_classifier, self.num_classes, self.dtype, self.kernel_init)

    def __call__(self, x, train):
        # TODO: Implement the construction and forward pass for RotNet. Check above link for hints.
        ############################## Your Code Starts Here #########################################
        # hint: Think carefully about the network structures: 
        #           1. The input should firstly be forwarded by Features module
        #           2. Then the result should be forwarded by Classifier module
        
        
        
        return None    
        ############################## Your Code Ends Here #########################################
        

def rotnet_constructor(model_arch):
    """ Creates a RotNet model whose structure is specified by model_arch.

    Args:
        model_arch (str):
            A string specifying the RotNet architecture. For example, 'rotnet3_feat2' means the RotNet contains three RotNetBlocks,
            and the output of the second RotNetBlock will be extracted for transfer learning.

    Returns:
        nn.Module: a RotNet model
    """
    cnn_channels = 64
    num_blocks, num_blocks_features = int(model_arch[6]), int(model_arch[12])
    num_classes = 4
    
    return RotNet(cnn_channels, num_blocks_features, num_blocks - num_blocks_features, num_classes)
