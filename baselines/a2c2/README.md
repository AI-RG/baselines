# A2C2: Experiments in RL

I am performing some RL experiments using an A2C learning algorithm. In particular, I have modified the implementation of the A2C in OpenAI's collection of baselines. There are two main modifications: 

- Criticality: I am interested in inducing behavior over time that is closer to criticality, i.e. perched between chaos and order.
- Capsules: Given the nice properties of capsule networks (e.g. good transfer learning), I was interested in exploring how well they would work in an RL setting.

## A2C + Criticality
A2C + criticality

This repository investigates the potential of Self-Organized Criticality (SOC) as a method to speed learning, in particular in a reinforcement learning context. Criticality is implemented practically by the addition of another loss term:

![Image of SOC loss term](https://github.com/AI-RG/rl-experiments/blob/master/lsoc.gif),

which penalizes the time-averaged hidden state *s* (element-wise). This penalty encourages the time average of each component of the state to change over the course of the averaging timescale, so that consistently large (near absolute magnitude 1) or small (near zero) time averages are penalized. One perspective on this approach is that it encourages exploration in the space of internal representations. By penalizing frozen components of hidden states, we incentivize models to take fuller advantage of their representational capabilities.

More information is avaialable in the `readme.md` file of `/criticality`.

## A2C + Capsules

This is a straightforward modification of the Capsule Network architecture implemented in Sabour-Frost-Hinton (2017) to function as an architecture in A2C. There is an encoding layer of capsules which feeds into a baseline value function estimate and also an additional capsule layer where the capsules represent discrete actions in a policy.

More information is available in the `readme.md` file of `/capsules`.

## How to use this code

The easiest way to use this code is to first download and install the OpenAI baselines repo from its main branch. Then, you can  insert the folder `a2c2` into the directory of `baselines` alongside `a2c`. To run the code, execute e.g. `python3 -m baselines.a2c2.run_atari --policy=caps`, which will train a Capsule Network policy on an Atari game (BreakOut, by default). To train with self-organized criticality, execute `python3 -m baselines.a2c2.run_atari --policy=lstm`; SOC is implemented by default when you select the lstm policy.

# A2C original readme

This repository is modified from a version of the A2C algorithm in OpenAI's collection of baselines.

- Original paper: https://arxiv.org/abs/1602.01783
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.a2c.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
