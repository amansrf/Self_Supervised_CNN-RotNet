## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install some Python packages; see [installation.md](installation.md) for instructions.
2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2022/blob/master/hw1/cs285/scripts/run_hw1.ipynb)

## Complete the code

Fill in sections marked with `TODO`. In particular, see
 - [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
 - [policies/MLP_policy.py](cs285/policies/MLP_policy.py)
 - [infrastructure/replay_buffer.py](cs285/infrastructure/replay_buffer.py)
 - [infrastructure/utils.py](cs285/infrastructure/utils.py)
 - [infrastructure/pytorch_util.py](cs285/infrastructure/pytorch_util.py)

Look for sections maked with `HW1` to see how the edits you make will be used.
Some other files that you may find relevant
 - [scripts/run_hw1.py](cs285/scripts/run_hw1.py) (if running locally) or [scripts/run_hw1.ipynb](cs285/scripts/run_hw1.ipynb) (if running on Colab)
 - [agents/bc_agent.py](cs285/agents/bc_agent.py)

See the homework pdf for more details.

## Run the code

Tip: While debugging, you probably want to keep the flag `--video_log_freq -1` which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!

If running on Colab, adjust the `#@params` in the `Args` class according to the commmand line arguments above.

### Section 1 (Behavior Cloning)
Command for problem 1:

```
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.
To generate videos of the policy, remove the `--video_log_freq -1` flag.

### Section 2 (DAgger)
Command for section 1:
(Note the `--do_dagger` flag, and the higher value for `n_iter`)

```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.

## Visualization the saved tensorboard event file:

You can visualize your runs using tensorboard:
```
tensorboard --logdir data
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

You can choose to visualize specific runs with a comma-separated list:
```
tensorboard --logdir data/run1,data/run2,data/run3...
```

If running on Colab, you will be using the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to do the same thing; see the [notebook](cs285/scripts/run_hw1.ipynb) for more details.

## Credit and References
This assignment was created starting with the following three base repositories:

1.  The original paper repo [FeatureLearningRotNet](https://github.com/gidariss/FeatureLearningRotNet). While this repo was a good first implementation, it is written in an extremely old version of PyTorch and so was not ideal to base the assignment on. It was however critical in understanding the concepts in the paper more deeply.
2.  The repo [Self_Supervised_CNN-RotNet](https://github.com/Poulinakis-Konstantinos/Self_Supervised_CNN-RotNet) in Tensorflow which was written more recently, allowed us to overcome some of the hurdles in the original paper repository. We based our JAX conversion on the framework provided here very heavily. Many thanks to the creators.
3.  The repo [Flax-ResNets](https://github.com/fattorib/Flax-ResNets) implements ResNets in both PyTorch and Flax and was very helpful in understanding the various Jax/Flax ideas to implement something on our own.

Other useful links and guides in no particular order:
1. The [Flax Getting Started Guide](https://flax.readthedocs.io/en/latest/getting_started.html) as the quintessential source for creating neural networks using Jax.
2. The [Flax Guide to Saving and Restoring Models](https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html)
3. The [Flax Guide to Transfer Learning](https://flax.readthedocs.io/en/latest/guides/transfer_learning.html), [Extracting Intermediates](https://flax.readthedocs.io/en/latest/guides/extracting_intermediates.html) and [this Kaggle Notebook example](https://www.kaggle.com/code/yashvi/transfer-learning-using-jax-flax/notebook). 

    *We would like to note that setting up transfer learning in Jax/Flax in this manner was extremely challenging and requires a lot to be desired in terms of how the framework is written.*
4. [How to think in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html): Especially to understand what JIT is and how to work with it. 

    *Pay close attention to static_argnums to save yourself a lot of pain later on!*
