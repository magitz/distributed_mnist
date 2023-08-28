# Multi-GPU Model Training with Keras

This tutorial is adapted from the TensorFlow tutorial here: [https://www.tensorflow.org/tutorials/distribute/keras](https://www.tensorflow.org/tutorials/distribute/keras)

To run this on HiPerGator, I requested an [Open onDemand](https://ood.rc.ufl.edu/) Jupyter session with:

* 4 cores
* 64 GB RAM
* 2 A100 GPUs: `gpus:a100:2`

This tutorial is intended to help get you up and running training models on HiPerGator and to show how simple it can be to scale things up as you exceed the capabilities of a single GPU. This tutorial also introduced [TensorBoard](https://www.tensorflow.org/tensorboard/get_started#:~:text=TensorBoard%20is%20a%20tool%20for,dimensional%20space%2C%20and%20much%20more.), a relatively easy method of tracking model training and your hyperparameter tuning experiments.

## Start with the Jupyter notebook

Get started running through the notebook and instructions here: [distributed_mnist.ipynb]

## Switch to using batch jobs

OK, now that you have that down, you might be thinking, "If this were a real model that needed multiple-GPUs to train, it would probably take a while. Plus, now that I know how to track my experiments in hyperparameter tuning, I want to run this lots of times using different hyperparameters...again longer than I want to sit in front of a Jupyter notebook!"

The good news is that we can easily convert this to run as a script--In Jupyter's File menu, select "Save and Export Notebook As..." > "Executable Script". Unfortunately, that downloads the notebook, so you have to upload it, but I've already done this for you.

This is a Python script that can be run with a simple command and will run through the whole script.

We can submit that to the HiPerGator batch system with another script.

1. Connect to HiPerGator with an ssh client: `ssh user@hpg.rc.ufl.edu`
1. Change directories to where is repo is located: e.g. `cd /blue/gms6029/$USER/distributed_mnist`
1. Open up and look at the [run_mnsit.sbatch](run_mnist.sbatch) script: `nano run_mnist.sbatch`
   * Edit your email address
1. To submit the script, type: `sbatch run_mnist.sbatch`
1. Look at the results


## Congrats!

You have now run parallel model training in a Jupyter notebook and via the command line!

I hope this helps you get started with doing more amazing things!
