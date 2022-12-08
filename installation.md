# Install Dependencies

 * **NOTE:** *Please look at the known bugs section below if you encounter a bug during installation, in case we did too and have a fix!*

There are three options:

## A. (`Recommended`) Install with conda:

1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

    This install will modify the `PATH` variable in your bashrc.
    You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

2. Create a conda environment that will contain python 3:
    ```
    conda create -n rotnet_jax python=3.8.10
    ```

3. activate the environment (do this every time you open a new terminal and want to run code):
    ```
    conda activate rotnet_jax
    ```

4. Install the requirements into this conda environment
    ```
    pip install -r requirements.txt
    ```
5. Install pytorch by following instructions [here](https://pytorch.org/get-started/locally/)

    **Remember to**:
    * Use the **conda installation**
    * **Choose GPU/CPU based on your system spec** 
    
    ### **NOTE:**
     **We only use pytorch to load the dataset and do some initial transforms as this is the recommended way by the JAX and FLAX community. This makes the dataloading convenient.**
     
     **Installing pytorch also has the added benefit of configuring CUDA in the environment in case you are using the GPU.**

6. Install jax:
    * For GPU (Make sure you meet requirements [here](https://github.com/google/jax#installation)):
        ```
            pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ```
    * For CPU:
        ```
            pip install --upgrade "jax[cpu]"
        ```

7. Install tensorflow: (**JAX/FLAX uses Tensorflow based checkpoint functions and so we must install it**)
    ```
        pip install tensorflow
    ```

This conda environment requires activating it every time you open a new terminal (in order to run code), but the benefit is that the required dependencies for this codebase will not affect existing/other versions of things on your computer. This stand-alone environment will have everything that is necessary.


## B. Install with virtualenv:

1. Install virtualenv, if you don't already have it, by running the following command:
    ```
    pip install virtualenv
    ```

2. Create a virtualenv environment that will contain python 3:
    ```
    virtualenv rotnet_jax -p python3.8.10
    ```

3. Activate the environment from the directory where it exists (do this every time you open a new terminal and want to run code):
    ```
    source rotnet_jax/bin/activate
    ```

4. Install the requirements into this virtualenv environment
    ```
    pip install -r requirements.txt
    ```

5. Install pytorch by following instructions [here](https://pytorch.org/get-started/locally/)

    **Remember to**:
    * Use the **pip installation**
    * **Choose GPU/CPU based on your system spec** 
    
    ### **NOTE:**
     **We only use pytorch to load the dataset and do some initial transforms as this is the recommended way by the JAX and FLAX community. This makes the dataloading convenient.**

6. Install jax:
    * For GPU (Make sure you meet requirements [here](https://github.com/google/jax#installation)):
        ```
            pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ```
    * For CPU:
        ```
            pip install --upgrade "jax[cpu]"
        ```

7. Install tensorflow: (**JAX/FLAX uses Tensorflow based checkpoint functions and so we must install it**)
    ```
        pip install tensorflow
    ```

## C. ( `NOT RECOMMENDED` ) Install with System Python:

Just follow the instructions in Section B above from Step 4.


# Known Bugs & Issues:
* There is an issue with the requirement for the msgpack-python package. If you any errors for the same when running the code, please try:
    ```
    pip install -U msgpack-python
    ```