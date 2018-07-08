# A2C2: A2C + criticality

This repository investigates the potential of Self-Organized Criticality (SOC) as a method to speed learning, in particular in a reinforcement learning context. Criticality is an intriguing property of dynamical systems. When a system is arranged so that its dynamics are poised on the threshold of chaos yet not chaotic, it is said to be *critical*. Particularly within the neuroscience community, there has been much interest in the role of criticality in information processing. Some research seeks to construct toy models that inch towards biological realism, and other research seeks to experimentally capture clear signals of criticality in actual brains. Meanwhile, other research seeks to better understand what benefits may be conferred by criticalty, and has concluded that there are likely many, including

- maximal dynamic range (sensitivity to the largest range of magnitudes of stimuli), 
- maximal information storage, and
- robustness to perturbations.

Criticality seems to balance the benefits of remembing the past with the advantages of continually learning new information. The upshot is that criticality appears adaptive and moreover actually present in neural systems. However, most dynamical systems require fine-tuning to stay at criticality, and it is not obvious how a system as complex and noisy as a brain could coordinate to maintain this finely tuned state of affairs. For this reason, there is growing interest in the possibility that neural systems have dynamical evolution that pushes them back towards criticality as they stray farther from it. Such phenomena are referred to as Self-Organized Criticality (SOC).

In these experiments, self-orgnanized criticality is implemented practically by the addition of a differntiable loss term:

![Image of SOC loss term](https://github.com/AI-RG/rl-experiments/blob/master/lsoc.gif),

which penalizes the time-averaged hidden state *s* (element-wise). This penalty encourages the time average of each component of the state to change over the course of the averaging timescale, so that consistently large (near absolute magnitude 1) or small (near zero) time averages are penalized. One perspective on this approach is that it encourages exploration in the space of internal representations. By penalizing frozen components of hidden states, we incentivize models to take fuller advantage of their representational capabilities.

<img src="https://github.com/AI-RG/baselines/blob/master/baselines/a2c2/criticality/assets/soc-penalty.png" width="400px"/>

This penalty is inspired by [1], which studied a toy model of self-organized criticality. In their setting, a Boolean network (with activations *a* in the two-element set {0, 1}) was evolved over time with the activations subject to a similar penalty. More precisely, the network was evolved to perch at criticality using the following procedure:

- the network dynamics were rolled out for *T* steps
- the time-average *A<sub>T</sub>(a)* of each activation *a* over the previous *T* steps was calculated
- The network was rewired to push each activation closer to criticality:
  - Nodes whose time-average activations exceeded a certain threshold and were frozen *on* (*A<sub>T</sub>(a) > t<sub>+</sub> < 1*) were subject to pruning of incoming edges, *decreasing* their probability of activation.
  - Nodes whose time-average activations fell below than a certain threshold and were frozen *off* (*A<sub>T</sub>(a) > t<sub>-</sub> < 1*) were randomly assigned new incoming edges, *increasing* their probability of activation.
  
Our procedure implements the same concept in a differentiable way. Moreover, we also experiment with penalizing the component-wise time-variance of the hidden state, given that unlike in [1], a hidden state activation may also be frozen at e.g. 1/2, since our activations are continuous-valued instead of Boolean.

## Experiments

We perform experiments on the `BreakOut` Atari environment of OpenAI's gym.

We also perform experiments with this SOC regularization on an LSTM language model. Specifically, we see whether the SOC penalty can help an LSTM learn the Penn Tree Bank corpus more rapidly.

## Details

### How to run this code

The command `python3 -m baselines.a2c2.run_atari --mode=soc` runs the algorithm for 40M frames = 10M timesteps, with additional command line options available; for a listing, use help (`-h`).

## Bibliography

1.  S. Bornholdt and T. Rohlf, "Topological Evolution of Dynamical Networks: Global Criticality from Local Dynamics" (2000)
2.  W. Shew and D. Plenz, "The Functional Benefits of Criticality in the Cortex" The Neuroscientist (2012)
3.  M. Munoz et al., "Griffiths phases on complex networks" (arXiv: 1009:0395)
4.  N. Bertschinger and T. Natschläger, "Real-time computation at the edge of chaos in recurrent neural networks" Neural Comput. 16 (2004)
5.  D. Markovic and C. Gros, "Power laws and self-organized criticality in theory and nature: comparison and review of different SOC processes" (2014)
6.  C. Meisel, et al. "Fading signatures of critical brain dynamics during sustained wakefulness in humans." J. Neurosci. 33 (2013)
7.  J. O'Brien, "A fundamental theory to model the mind." Quanta (2014).
8.  J. Hesse and T. Gross. "Self-organized criticality as a fundamental property of neural systems." Front. Syst. Neorosci. (23 Sep 2014)
9.  S. Johnson, J. Marro, and J. Torres. "Robust short-term memory without synaptic learning." PlosOne (2012) [NB: This S. Johnson is not me (!). I like his style, though.]
10.  P. Moretti and M. Munoz. "Griffiths phases and the stretching of criticality in brain networks" Nature Communications 4 (2013).

## A2C original

This repository is modified from a version of the A2C algorithm in OpenAI's collection of baselines.

- Original paper: https://arxiv.org/abs/1602.01783
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.a2c.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
