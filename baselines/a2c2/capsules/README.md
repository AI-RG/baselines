# A2C with a Capsule Network Policy

Capsule networks were developed by in [] and shown to be quite good at transfer learning in []. The idea, at its core, is to encode the probability that certain entities are present in the input by the magnitude of their vector representations. This approach seems quite natural to apply in the context of reinforcement learning. In particular, in discrete action spaces, one can model each potential action as a capsule.

## Capsule nets in general

In [], the concept of a "capsule" was introduced. The idea is to 

[] used an architecture as shown in the figure in order to classify digits in MNIST and affine-MNIST, in which digits are randomly translated and one digit together with other digit fragments may be present in a single image.

## Our architecture

The present architecture combines elements of a widely used A2C actor-critic architecture with the basic components of the CapsuleNet architecture. Namely, the initial processing is identical (except in hyperparameters) to the network in []; however, the final capsule layer encodes aspects of the scene that will be processed further. This single layer is connected to both:
- a one-layer FFN with one-dimensional output, which produces a value-function estimate V(s); and
- a final capsule layer with as many capsules as discrete actions, producing a policy.

A schematic of the architecture is shown below:

<img src="https://github.com/AI-RG/baselines/blob/master/baselines/a2c2/capsule-policy.png" alt="caps-policy" width="800px"/>


## Experiments

## Possible extensions

## Details

## Bibliography

Original A3C paper:
[1] V. Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (arXiv:1602.01783)

[2] Original Capsule paper:
S. Sabour, N. Front, and G. Hinton, "Dynamic Routing Between Capsules" (arXiv: 1710.09829)

[3] Capsules are good at transfer learning:
A. Gritsevskiy and M. Korablyov, "Capsule networks for low-data transfer learning" (arXiv: 1804.10172)
