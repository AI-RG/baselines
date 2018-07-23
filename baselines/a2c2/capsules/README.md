# A2C with a Capsule Network Policy

Capsule networks were developed by in [1] and shown to be quite good at transfer learning in [2]. The idea, at its core, is to encode the probability that certain entities are present in the input by the magnitude of their vector representations. This approach seems quite natural to apply in the context of reinforcement learning. In particular, in discrete action spaces, one can model each potential action as a capsule.

## Capsule nets in general

In [1], the concept of a "capsule" was introduced. The idea is to represent features using vectors; then, the norm of a given vector encodes the probability that its feature exists. Each such vector was dubbed a *capsule* in [1]. This extra structure permits information to be routed between capsules of successive layers in a more nuanced way than is possible between simple nodes in vanilla MLPs or CNNs. Namely, the effective weight between capsule *j* of layer *L* and capsule *i* of layer *L+1* is influenced by the dot product *v<sub>i</sub>.v<sub>j</sub>*. ([1] presents the full details of this process quite clearly, so I won't reproduce them here.)

[1] used an architecture as in the figure to classify digits in MNIST and multi-MNIST, single images of which contain two (overlapping) digits.

<img src="https://github.com/AI-RG/baselines/blob/master/baselines/a2c2/capsule-policy-original.png" alt="orig-caps-policy" width="800px"/>

## Our architecture

The present architecture combines elements of a widely used A2C actor-critic architecture [3] with the basic components of the CapsuleNet architecture. Namely, the initial processing is identical (except in hyperparameters) to the network in [1]; however, the final capsule layer encodes aspects of the scene that will be processed further. This single layer is connected to both:
- a one-layer MLP with one-dimensional output, which produces a value-function estimate V(s); and
- a final capsule layer with as many capsules as discrete actions, producing a policy.

A schematic of the architecture is shown below:

<img src="https://github.com/AI-RG/baselines/blob/master/baselines/a2c2/capsule-policy.png" alt="caps-policy" width="800px"/>

## Experiments

The main drawback of the Capsule Policy as implemented above is that it takes longer to converge to a good value than e.g. a CNN baseline. This is true both in terms of episodes, and wall time, where the CapsNet is approximately 5-6 times slower than a CNN. (This could be a coincidence, but there are 2 routing stages with 3 iterations each.) However, an advantage of the Capsule Policy is that it exhibits less variance in learning. Moreover, unlike the CNN policy, the CapsNet did not demonstrate any unusual dips in explianed variance throughout its policy improvement (as seen in the figure).

The longer training time of our architecture is not unusual. A quick count of parameters shows that the CapsNet has 7,981,697 parameters in contrast to the 1,680,549 of the CNN. A smaller architecture could have been chosen, but it appears that this excess capacity is still trainable in practice and may facilitate better multi-task learning at the expense of slower initial training.

<img src="https://github.com/AI-RG/baselines/blob/master/baselines/a2c2/capsules/ev_caps.png" alt="caps-ev" width="650px"/>



## Possible extensions

### Transfer learning

The more modular organization of casule networks lends them to transfer learning. As such, it would be interesting to see how well a pretrained (e.g. Reptile [4]) CapsuleNet could adapt to different environments, e.g. different Atari games. This is the subject of ongoing work.

### Recurrence

There is already an element of recurrence in CapsNets, namely the dynamic routing procedure (see algorithm 1 in [1]), which iteratively estimates how much weight *c<sub>ij</sub>* to assign capsule *j* of layer *L* in determining the value of capsule *i* of layer *L+1*. The most straightforward way to add recurrence to CapNets is simply to refrain from resetting *c<sub>ij</sub>*  (or more properly *b<sub>ij</sub>*) before each new observation; this amounts to adding a prior of the previous observations' routings to the current one.

The addition of a GRU-like gated update mechanism would represent a larger modification in the direction of recurrence. Using the final layer ("Encoded Caps") as the hidden state, one could form a gated update with new input *x' = PrimaryCaps(x)* the outupt of PrimaryCaps. Since the `squash()` nonlinearity normalizes inputs, it is already well suited for use in a recurrent architecture.

Experiments in this direction are ongoing. 

## Details

### How to run this code

The command `python3 -m baselines.a2c2.run_atari --policy=CapsulePolicy` runs the algorithm for 40M frames = 10M timesteps, with additional command line options available; for a listing, use help (`-h`).

By default, the algorithm runs on the Atari game Breakout (gym environment `BreakoutNoFrameSkip-v4`), with four consecutive stacked frames of (grayscale) pixels as input. Moreover, 16 different environments are run in parallel for 5 time steps each before synchronously updating the network's parameters. This default behavior has not been modified from OpenAI's original implementation.

## Bibliography

- [1] S. Sabour, N. Front, and G. Hinton, "Dynamic Routing Between Capsules" (arXiv: 1710.09829)
- [2] A. Gritsevskiy and M. Korablyov, "Capsule networks for low-data transfer learning" (arXiv: 1804.10172) 
- [3] V. Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (arXiv:1602.01783)
- [4] A. Nichol and J. Shulman, "Reptile: A Scalable Metalearning Algorithm" (arXiv: 1803.02999)
