# ECEN743-SP24-HW02

## Overview

1. You have to submit a report in `HTML` and your code to Canvas.
2. Put all your files (`HTML` report and code) into a **single compressed folder** named `Lastname_Firstname_A2.zip`.
3. If you are using Jupyter Notebook, you can export it in `HTML` by going through the top toolbar:
    ```
    "File -> Save and Export Notebook As... -> HTML"
    ```
    If you are using Google Colab, you might need to do some extra steps to produce an `HTML` report. Please Google for "how to convert ipynb notebook to HTML in Google Colab?".
4. This homework is self-containted in one Jupyter notebook. In your `zip`, we expect only your `HTML` report and **one** Jupyter notebook.

## Installation Intructions

1. If you wish to complete this assignment locally (not on Google Colab), you need to install Jupyter Notebook. You can do  
    ```
    pip install jupyter notebook
    ```
2. In this assignment, you will play around with the famous `FrozenLake` environment. Please install Gymnasium (you can read more about Gymnasium [here](https://gymnasium.farama.org/)).
    ```
    pip install gymnasium
    ```
3. It is strongly advised that you learn how to use virtual environment for Python. It creates an isolated environment from the system Python or other Python releases you have installed system-wide. It helps you manage Python packages in a clean fashion and allow you to only install necessary packages for particular projects. An exemplary, lightweight virtual environment module is `venv` [(link)](https://docs.python.org/3/library/venv.html). Your python distribution is likely to include it by default. If not, for example on Ubuntu, you can install it by
    ```
    sudo apt-get install python3-venv
    ```

## Assignment
In this assignment, you will implement a cononical model-free reinforcement learning algorithm. The term *model-free* means that the learner (you) does not have or do not explicitly estimate the model (transition probabilities) of the environment. You will still work on the `FrozenLake` environment from Gymnasium [(Link)](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).

1. **Tabular Q-Learning:** Implement the tabular Q-Learning algorithm on the FrozenLake environment.  
    (a). Plot $G_k$, where $G_k$ is the cumulative reward obtained in episode $k$. Use a sliding window averaging to obtain smooth plots.  
    (b). Plot $\lVert Q_k-Q^* \rVert$, where $Q_k$ is the Q-value during the $k^{\mathrm{th}}$ iteration, and $Q^*$ is the optimal Q-value function obtained from QVI. Note that the optimal Q-value function will be given to you, and you do not need to copy $Q^*$ from your Assignment 1 submission.   
    (c). What is the policy and Q-value function obtained at the end of the learning? Are you able to learn the optimal policy?  

2. **Behavior Policy:** Implement tabular Q-learning with  a uniformly random policy (where each action is taken with equal probability) as the behavior policy. Compare the convergence with the $\epsilon$-greedy exploration approach. Explain your observations and inference. Can you implement a better behavior policy and show its effectiveness?  

3. **TD-Learning:** Consider the following polices: (i) the optimal policy obtained from QVI and (ii) a uniformly random policy where each action is taken with equal probability. Learn the value of these policies using:
    (a). Monte Carlo (MC) Learning 
    (b). Temporal Difference (TD) Learning
    (c). What are the trade-offs of between MC vs. TD?