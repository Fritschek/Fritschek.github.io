---
layout: single
title: "From Diffusion to Drifting: Generative Modeling as Learned Distributional Transport"
date: 2026-05-23
tags: [generative-models, diffusion, optimal-transport, sinkhorn]
excerpt: "A note on drifting models, Wasserstein gradient flows, and one-step generators as training-time distributional transport."
---

*A technical note on Deng et al.'s drifting models, the Wasserstein-gradient-flow
interpretation, and the recent W-Flow construction.*

---

## 1. Pushforwards

The same mathematical object appears across diffusion models, flow matching,
drifting models, and W-Flow: transport from a simple reference distribution to
the data distribution. The computational question is where this transport is
carried out.

Diffusion and score-based models require integration of learned reverse-time
dynamics during sampling [[1]](#ref-sohl), [[2]](#ref-song). Flow matching uses
velocity-field regression between distributions [[4]](#ref-lipman). Drifting
updates the generator's pushforward distribution during training and keeps
sampling to one network evaluation [[5]](#ref-deng). W-Flow represents the
training-time transport as a Wasserstein gradient flow based on the Sinkhorn
divergence [[9]](#ref-han).

Let $z \sim \rho_0$ denote a latent variable drawn from a simple reference
distribution, and let $G_\theta$ denote the generator. The induced model
distribution is

$$
\rho_\theta = (G_\theta)_\# \rho_0.
$$

This is the distribution of generated samples. Training changes the generator
until

$$
\rho_\theta \approx \rho_{\mathrm{data}}.
$$

Two modeling choices remain:

1. Which distributional path is used?
2. Is that path evaluated at inference time or absorbed during training?

## 2. Diffusion: dynamics at inference time

A continuous-time diffusion model starts with a noising process

$$
dX_t = f(t,X_t)\,dt + g(t)\,dW_t, \qquad X_0 \sim \rho_{\mathrm{data}}.
$$

For a scalar diffusion coefficient $g(t)$, the density evolves according to
the Fokker-Planck equation

$$
\partial_t \rho_t
=
-\nabla \cdot (f \rho_t)
+
\frac{1}{2} g(t)^2 \Delta \rho_t.
$$

If the noising process is run long enough, the data distribution approaches a
simple reference distribution. Sampling is implemented by reverse-time
integration. With the usual reverse-time convention, the reverse SDE contains
the score

$$
\nabla_x \log \rho_t(x),
$$

and its drift has the schematic form

$$
f(t,x) - g(t)^2 \nabla_x \log \rho_t(x).
$$

Different reverse-time parametrizations lead to different signs in this display.
The learned score field is evaluated repeatedly during sampling. A diffusion
model combines a map from noise with a learned numerical procedure.

This costs sampling time. High-quality samples usually require repeated network
evaluations.

## 3. Transport viewpoints

Transport formulations start from paths between probability distributions.
Schrödinger bridges specify a stochastic process with fixed endpoint
marginals,

$$
X_0 \sim \rho_0, \qquad X_1 \sim \rho_{\mathrm{data}},
$$

and whose path measure $P$ is close to a reference process $R$:

$$
\min_P \mathrm{KL}(P\|R)
\quad
\text{subject to}
\quad
P_0 = \rho_0,\qquad P_1 = \rho_{\mathrm{data}}.
$$

This is dynamic, entropy-regularized optimal transport [[3]](#ref-debortoli).
Flow matching fits a velocity field that carries samples along a prescribed
probability path [[4]](#ref-lipman).

The emphasis shifts from denoising to transport between distributions.

## 4. Wasserstein gradient flows

Optimal transport gives a geometry on probability measures. A moving
distribution in this geometry satisfies the continuity equation

$$
\partial_t \rho_t + \nabla \cdot (\rho_t v_t) = 0.
$$

It encodes conservation of mass under the velocity field $v_t$. Given an energy
functional $\mathcal{E}(\rho)$ with small values near
$\rho_{\mathrm{data}}$, the Wasserstein gradient descent equation is

$$
\partial_t \rho_t
=
\nabla \cdot
\left(
  \rho_t \nabla \frac{\delta \mathcal{E}}{\delta \rho}(\rho_t)
\right).
$$

Equivalently, the particle velocity is

$$
v_t(x)
=
-\nabla
\frac{\delta \mathcal{E}}{\delta \rho}(\rho_t)(x).
$$

After the variational derivative and the spatial gradient, a distributional
discrepancy yields a velocity field for generated samples.

## 5. Drifting models

Deng et al. describe iteration $k$ through the pushforward distribution of the
current generator [[5]](#ref-deng):

$$
\rho_k = (G_{\theta_k})_\# \rho_0.
$$

Generated samples and real samples are used to estimate a drifting field. A
schematic population form is

$$
V(x)
\approx
\int K(x,y)(y-x)\,d\rho_{\mathrm{data}}(y)
-
\int K(x,y)(y-x)\,d\rho_{\mathrm{model}}(y).
$$

The first term pulls updates towards data regions. The second term corrects for
the model's own mass and adds a repulsive/diversity component. This display is
only a population sketch. The algorithm uses finite samples, kernel
normalization, and implementation choices that matter. At the population level,
the field has the fixed-point condition

$$
V(\rho,\rho_{\mathrm{data}}) = 0
\quad \text{when} \quad
\rho = \rho_{\mathrm{data}}.
$$

A schematic training step is

$$
x = G_{\theta_k}(z), \qquad
x^+ = x + \eta V(x),
$$

followed by regression onto the drifted target $x^+$:

$$
\theta_{k+1}
\approx
\arg\min_\theta
\mathbb{E}_{z \sim \rho_0}
\left\|G_\theta(z) - x^+\right\|^2.
$$

The iterative motion occurs while training the generator. After training,
sampling reduces to

$$
z \sim \rho_0, \qquad x = G_\theta(z).
$$

Drifting moves the distribution during training and amortizes this motion into
the generator parameters.

Gretton et al. separate the idealized Wasserstein-gradient-flow interpretation
from the implemented drifting algorithm [[6]](#ref-gretton). The implemented
method resembles a fixed-point procedure, and its convergence properties need
not match those of the corresponding Sinkhorn-gradient-flow construction.

## 6. W-Flow and the Sinkhorn energy

Han et al. make the gradient-flow construction concrete by using the Sinkhorn
divergence to the data distribution as the driving energy [[9]](#ref-han).
Their derivation begins with the squared Wasserstein distance

$$
W_2^2(\rho,\nu)
=
\inf_{\pi \in \Pi(\rho,\nu)}
\int \|x-y\|^2\,d\pi(x,y),
$$

where $\Pi(\rho,\nu)$ is the set of couplings with marginals $\rho$ and
$\nu$. Exact optimal transport is expensive, so the transport problem is often
regularized by entropy [[7]](#ref-cuturi):

$$
\mathrm{OT}_\varepsilon(\rho,\nu)
=
\inf_{\pi \in \Pi(\rho,\nu)}
\int c(x,y)\,d\pi(x,y)
+
\varepsilon\,\mathrm{KL}(\pi \| \rho \otimes \nu).
$$

Entropic regularization leads to Sinkhorn iterations and introduces a bias. The
Sinkhorn divergence removes this bias by subtracting self-costs [[8]](#ref-feydy):

$$
S_\varepsilon(\rho,\nu)
=
\mathrm{OT}_\varepsilon(\rho,\nu)
-
\frac{1}{2}\mathrm{OT}_\varepsilon(\rho,\rho)
-
\frac{1}{2}\mathrm{OT}_\varepsilon(\nu,\nu).
$$

With the Sinkhorn divergence, the energy is

$$
\mathcal{E}(\rho) = S_\varepsilon(\rho,\rho_{\mathrm{data}}).
$$

The corresponding Wasserstein gradient flow reads

$$
\partial_t \rho_t
=
\nabla \cdot
\left(
  \rho_t
  \nabla
  \frac{\delta S_\varepsilon(\rho_t,\rho_{\mathrm{data}})}{\delta \rho}
\right).
$$

The induced particle velocity is

$$
v_t(x)
=
-
\nabla
\frac{\delta S_\varepsilon(\rho_t,\rho_{\mathrm{data}})}{\delta \rho}(x).
$$

The algorithm has two levels:

1. A distributional path from the reference distribution to the data
   distribution is defined by this Wasserstein gradient flow.
2. A static generator is fitted to approximate the endpoint of that path.

The shared feature with drifting is training-time distributional transport.
W-Flow adds a fixed energy, an optimal-transport interpretation of the induced
velocity, and finite-sample analysis against the continuous-time distributional
dynamics [[9]](#ref-han).

## 7. Relation to nearby model classes

| Framework | Main object | Training target | Sampling |
|---|---|---|---|
| Diffusion / score model [[1]](#ref-sohl), [[2]](#ref-song) | Reverse-time SDE or probability-flow ODE | Score $\nabla \log \rho_t$ | Iterative |
| Schrödinger bridge [[3]](#ref-debortoli) | Entropy-regularized path measure | Bridge dynamics | Usually iterative |
| Flow matching [[4]](#ref-lipman) | Velocity field between distributions | Conditional vector field | ODE integration, unless distilled |
| GAN [[10]](#ref-goodfellow) | Static pushforward map | Adversarial divergence proxy | One step |
| Drifting [[5]](#ref-deng) | Training-time pushforward evolution | Drifting field / fixed point | One step |
| W-Flow [[9]](#ref-han) | Wasserstein gradient flow | Sinkhorn-divergence energy descent | One step |

The table separates methods by where the learned dynamics are evaluated.
Diffusion and flow matching typically retain learned dynamics at sampling time.
GANs, drifting models, and W-Flow use one-step generation. Drifting and W-Flow
retain a distributional-transport interpretation and end with a static
generator.

## 8. Compact derivation

A generator defines a distribution:

$$
\rho_\theta = (G_\theta)_\# \rho_0.
$$

An energy compares the generated distribution with the data
distribution:

$$
\mathcal{E}(\rho) = D(\rho,\rho_{\mathrm{data}}).
$$

For W-Flow the discrepancy is the Sinkhorn divergence,

$$
D(\rho,\rho_{\mathrm{data}})
=
S_\varepsilon(\rho,\rho_{\mathrm{data}}).
$$

Wasserstein steepest descent defines the distributional evolution:

$$
\partial_t \rho_t
=
\nabla \cdot
\left(
  \rho_t \nabla \frac{\delta \mathcal{E}}{\delta \rho}
\right).
$$

Equivalently, the particle dynamics satisfy

$$
\frac{dX_t}{dt}
=
v_t(X_t),
\qquad
v_t
=
-
\nabla \frac{\delta \mathcal{E}}{\delta \rho}.
$$

The generator update then absorbs this motion:

$$
G_{\theta_{k+1}}(z)
\approx
G_{\theta_k}(z)
+
\eta v_k(G_{\theta_k}(z)).
$$

After training, no trajectory has to be integrated:

$$
z \sim \rho_0, \qquad x = G_\theta(z).
$$

## 9. Takeaway

One-step generation is relevant both computationally and mathematically. The
mathematical point is its connection to a controlled evolution of probability
measures.

GAN-like generators are fast, but their training signal is less directly tied
to explicit distributional dynamics. Diffusion models provide such a signal
through repeated inference-time evaluations. Drifting and W-Flow retain the
distributional signal and shift the computation into training.

The main open questions are about the geometry and the finite-sample estimates:
Which energy gives the right motion? How stable is the induced velocity when it
is estimated from batches? How expressive must $G_\theta$ be to absorb the
flow? And when does the fixed point of the training dynamics actually coincide
with the data distribution?

The result is an amortized form of distributional transport. The flow is learned
during training and represented by a single generator evaluation at inference.

## References

1. <span id="ref-sohl"></span>J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, ICML, 2015. [arXiv](https://arxiv.org/abs/1503.03585)

2. <span id="ref-song"></span>Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR, 2021. [arXiv](https://arxiv.org/abs/2011.13456)

3. <span id="ref-debortoli"></span>V. De Bortoli, J. Thornton, J. Heng, and A. Doucet, *Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling*, NeurIPS, 2021. [arXiv](https://arxiv.org/abs/2106.01357)

4. <span id="ref-lipman"></span>Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, *Flow Matching for Generative Modeling*, ICLR, 2023. [arXiv](https://arxiv.org/abs/2210.02747)

5. <span id="ref-deng"></span>M. Deng, H. Li, T. Li, Y. Du, and K. He, *Generative Modeling via Drifting*, arXiv:2602.04770, 2026. [arXiv](https://arxiv.org/abs/2602.04770)

6. <span id="ref-gretton"></span>A. Gretton, L. K. Wenliang, A. Galashov, J. Thornton, V. De Bortoli, and A. Doucet, *On the Wasserstein Gradient Flow Interpretation of Drifting Models*, arXiv:2605.05118, 2026. [arXiv](https://arxiv.org/abs/2605.05118)

7. <span id="ref-cuturi"></span>M. Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*, NeurIPS, 2013. [paper](https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport)

8. <span id="ref-feydy"></span>J. Feydy, T. Séjourné, F.-X. Vialard, S.-i. Amari, A. Trouvé, and G. Peyré, *Interpolating between Optimal Transport and MMD using Sinkhorn Divergences*, AISTATS, 2019. [PMLR](https://proceedings.mlr.press/v89/feydy19a.html)

9. <span id="ref-han"></span>J. Han, P. Li, Q. Guo, R. Xu, S. Ermon, and E. J. Candès, *One-Step Generative Modeling via Wasserstein Gradient Flows*, arXiv:2605.11755, 2026. [arXiv](https://arxiv.org/abs/2605.11755)

10. <span id="ref-goodfellow"></span>I. Goodfellow et al., *Generative Adversarial Nets*, NeurIPS, 2014. [arXiv](https://arxiv.org/abs/1406.2661)
