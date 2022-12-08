## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install some Python packages; see [installation.md](installation.md) for instructions.
2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2022/blob/master/hw1/cs285/scripts/run_hw1.ipynb)

## Complete the code

Fill in sections marked with `TODO`. In particular, see
 - TODO

See the homework pdf for more details.

## Run the code

    ### TODO

## Credit and References
This assignment was created starting with the following three base repositories:

1.  **The original paper repo [FeatureLearningRotNet](https://github.com/gidariss/FeatureLearningRotNet)**. While this repo was a good first implementation, it is written in an extremely old version of PyTorch and so was not ideal to base the assignment on. It was however critical in understanding the concepts in the paper more deeply.
2.  **The repo [Self_Supervised_CNN-RotNet](https://github.com/Poulinakis-Konstantinos/Self_Supervised_CNN-RotNet)** in Tensorflow which was written more recently, allowed us to overcome some of the hurdles in the original paper repository. We based our JAX conversion on the framework provided here very heavily. Many thanks to the creators.
3.  **The repo [Flax-ResNets](https://github.com/fattorib/Flax-ResNets)** implements ResNets in both PyTorch and Flax and was very helpful in understanding the various Jax/Flax ideas to implement something on our own.

Other useful links and guides in no particular order:
1. **The [Flax Getting Started Guide](https://flax.readthedocs.io/en/latest/getting_started.html)** as the quintessential source for creating neural networks using Jax.
2. **The [Flax Guide to Saving and Restoring Models](https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html)**
3. **The [Flax Guide to Transfer Learning](https://flax.readthedocs.io/en/latest/guides/transfer_learning.html)**, [Extracting Intermediates](https://flax.readthedocs.io/en/latest/guides/extracting_intermediates.html) and [this Kaggle Notebook example](https://www.kaggle.com/code/yashvi/transfer-learning-using-jax-flax/notebook). 

    *We would like to note that setting up transfer learning in Jax/Flax in this manner was extremely challenging and requires a lot to be desired in terms of how the framework is written.*
4. **[How to think in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)**: Especially to understand what JIT is and how to work with it. 

    *Pay close attention to static_argnums to save yourself a lot of pain later on!*

# Contact
* Tianlun Zhang | [Email @berkeley.edu](mailto:ztl1998@berkeley.edu)
* Jackson Gao | [Email @berkeley.edu](mailto:xgao@berkeley.edu)
* Jaewon Lee | [Email @berkeley.edu](mailto:jwonlee@berkeley.edu)
* Aman Saraf | [Email @berkeley.edu](mailto:aman_saraf@berkeley.edu)