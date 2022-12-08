from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import flax.linen as nn

ModuleDef = Any
dtypedef = Any

class PredNetBlock(nn.Module):
    """ The basic building block for PredNet.

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
            TODO: Implement forward Pass for PredNetBlock. 
            Each block should contains:
                1. a convolutional layer
                2. a batch norm layer. hint: use the norm function defined above
                3. a relu layer
            hint: Check out how to define a module using nn.compact here: https://flax.readthedocs.io/en/latest/guides/setup_or_nncompact.html
        """
        ############################## Your Code Starts Here #########################################
        
    
    
       ############################## Your Code Ends Here #########################################

class Classifier(nn.Module):
    """ Classifier module of a PredNet. A Classifier module consist of self.num_blocks of PredNetBlocks

        Args:
            cnn_channels (int): number of channels for all Convolutional layers (We keep them the same for simplicity)
            num_blocks (int): number of PredNetBlocks to include in Features submodule
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
        # TODO: complete the forward pass by ddding self.num_blocks of PredNetBlocks
        ############################## Your Code Starts Here #########################################
        # hint: We did something similar while implementing RotNet
            
        ############################## Your Code Ends Here #########################################
        x = nn.Dense(features=self.num_classes, dtype=self.dtype, kernel_init=self.kernel_init)(x)
        return x

class PredNet(nn.Module):
    """ The PredNet class contains two submodules: backbone and Classifier.
        The output of Features submodule will be extracted for transfer learning.

    Args:
        cnn_channels (int): number of channels for all Convolutional layers (We keep them the same for simplicity)
        num_blocks_features (int): number of RotNetBlocks to include in Features submodule
        num_blocks_classifier (int): number of RotNetBlocks to include in Classifier submodule
        num_classes (int): number of classes (used in the last layer of Classifier submodule)
        dtype (dtypedef): the type of data the model will be dealing with
        kernel_init (Callable): the type of initialization to use for initializing the model params
    """
    backbone: nn.Module
    cnn_channels: int
    num_blocks_classifier: int
    num_classes: int
    dtype: dtypedef = jnp.float32
    kernel_init: Callable = nn.initializers.glorot_uniform()

    def setup(self):
        # TODO: setup the Classifier module of PredNet
        ############################## Your Code Starts Here #########################################
        # hint: We did something similar while implementing RotNet
        pass  
            
        ############################## Your Code Ends Here #########################################

    def __call__(self, x, train):
        # TODO: Implement the construction and forward pass for RotNet
        ############################## Your Code Starts Here #########################################
        # hint: Think carefully about the network structures: 
        #           1. The input should firstly be forwarded by backbone module
        #           2. Then the result should be forwarded by Classifier module
        return None    
            
        ############################## Your Code Ends Here #########################################
        

def prednet_constructor(model_arch, backbone):
    """ Creates a PredNet model with a given backbone module, whose structure is specified by model_arch.

    Args:
        backbone: 
        
        model_arch (str):
            A string specifying the RotNet architecture. For example, 'rotnet3_feat2' means the RotNet contains three RotNetBlocks,
            and the output of the second RotNetBlock will be extracted for transfer learning.

    Returns:
        nn.Module: a PredNet model
    """
    cnn_channels = 128
    num_blocks_classifier = int(model_arch[7])
    num_classes = 10
    
    # TODO: return a PredNet with above arguments
    ############################## Your Code Starts Here #########################################
    # hint: We did something similar while implementing RotNet
    
    return ...        
    ############################## Your Code Ends Here #########################################