---
layout: single
title: "From Diffusion to Drifting: Generative Modeling as Learned Distributional Transport"
date: 2026-05-23
tags: [generative-models, diffusion, optimal-transport, sinkhorn]
excerpt: "A note on drifting models, Wasserstein gradient flows, and why one-step generators can be viewed as training-time distributional transport."
---

*A short note on Deng et al.'s drifting models, the Wasserstein-gradient-flow
interpretation, and the recent W-Flow construction.*

---

## 1. A pushforward viewpoint

Diffusion models, flow matching, drifting models, and W-Flow can all be read as
ways of moving probability mass from a simple reference distribution to the data
distribution. The location of this movement distinguishes the methods.

Diffusion and score-based models learn dynamics that are still integrated at
sampling time [[1]](#ref-sohl), [[2]](#ref-song). Flow matching learns a velocity
field between distributions [[4]](#ref-lipman). Drifting models move the
generator's pushforward distribution during training and then sample with one
network evaluation [[5]](#ref-deng). W-Flow makes a related idea explicit by
choosing a Wasserstein gradient flow, instantiated with the Sinkhorn divergence,
and then compressing that flow into a one-step generator [[9]](#ref-han).

The common object is the pushforward distribution of the generator:

$$
\rho_\theta = (G_\theta)_\# \rho_0.
$$

Here $z$ is drawn from the reference distribution, the generator maps it into
data space, and the displayed pushforward is the distribution of generated
samples. Generative modeling is then the problem of changing the generator until

$$
\rho_\theta \approx \rho_{\mathrm{data}}.
$$

The pushforward view separates two questions that are often mixed together:

1. Which path should the distribution follow?
2. Should that path be followed at inference time, or absorbed during training?

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

If the noising process is run long enough, the data distribution is pushed
towards a simple reference distribution. Sampling then means running the process
backwards. With the usual reverse-time convention, the reverse SDE contains the
score

$$
\nabla_x \log \rho_t(x),
$$

and its drift has the schematic form

$$
f(t,x) - g(t)^2 \nabla_x \log \rho_t(x).
$$

The sign convention depends on whether one writes the reverse process with
decreasing time $t$ or with a new increasing time variable. The important
point here is simpler: the learned score field is evaluated repeatedly during
sampling. A diffusion model combines a map from noise with a learned numerical
procedure.

The price of this construction is inference-time computation: high-quality
sampling requires repeated network evaluations.

## 3. Transport viewpoints

A complementary reading of diffusion-like methods starts from paths between
probability distributions.

In a Schrödinger bridge, one looks for a stochastic process whose endpoint
marginals are fixed,

$$
X_0 \sim \rho_0, \qquad X_1 \sim \rho_{\mathrm{data}},
$$

and whose path measure $P$ is close to a reference process $R$:

$$
\min_P \mathrm{KL}(P\|R)
\quad
\text{subject to}
\quad
P_0 = \rho_0,\; P_1 = \rho_{\mathrm{data}}.
$$

This is the dynamic, entropy-regularized optimal-transport view of the problem
[[3]](#ref-debortoli). Flow matching gives another transport formulation: learn a
velocity field that carries samples along a prescribed probability path
[[4]](#ref-lipman).

These views change the language. The model becomes a mechanism for moving mass
between distributions.

## 4. Wasserstein gradient flows

Optimal transport gives a geometry on probability measures. In that geometry, a
moving distribution satisfies the continuity equation

$$
\partial_t \rho_t + \nabla \cdot (\rho_t v_t) = 0.
$$

This says that the density changes because particles move with velocity
$v_t$. Now choose an energy functional $\mathcal{E}(\rho)$ that should be
small when $\rho$ is close to $\rho_{\mathrm{data}}$. The Wasserstein
gradient descent equation is

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

The useful step is that a distributional discrepancy induces a direction in
which generated samples should move.

## 5. Drifting models

Deng et al. start from the pushforward distribution of the current generator
[[5]](#ref-deng):

$$
\rho_k = (G_{\theta_k})_\# \rho_0.
$$

At training step $k$, generated samples and real samples are used to estimate
a drifting field. In a schematic population form, this looks like

$$
V(x)
\approx
\int K(x,y)(y-x)\,d\rho_{\mathrm{data}}(y)
-
\int K(x,y)(y-x)\,d\rho_{\mathrm{model}}(y).
$$

The first term pulls samples towards data regions. The second term subtracts the
model's own mass and acts as a repulsive/diversity term. The displayed
expression is a simplified population picture; the actual algorithm uses finite
samples, kernel normalization, and implementation details that matter. It
captures the basic fixed-point idea:

$$
V(\rho,\rho_{\mathrm{data}}) = 0
\quad \text{when} \quad
\rho = \rho_{\mathrm{data}}.
$$

One training step can then be read as

$$
x = G_{\theta_k}(z), \qquad
x^+ = x + \eta V(x),
$$

followed by a regression step that trains the generator to produce $x^+$:

$$
\theta_{k+1}
\approx
\arg\min_\theta
\mathbb{E}_{z \sim \rho_0}
\left\|G_\theta(z) - x^+\right\|^2.
$$

So the iterative motion happens while training the generator. After training,
sampling is just

$$
z \sim \rho_0, \qquad x = G_\theta(z).
$$

In drifting, the distributional motion occurs during training and is amortized
into the generator parameters.

Gretton et al. make an important distinction here [[6]](#ref-gretton). An
idealized version of drifting can be interpreted through Wasserstein gradient
flows. The implemented algorithm of Deng et al. resembles such a fixed-point
procedure, with separate properties from the corresponding
Sinkhorn-gradient-flow construction.

## 6. W-Flow and the Sinkhorn energy

W-Flow makes the energy functional explicit [[9]](#ref-han). The starting point is
the squared Wasserstein distance

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

This regularization leads to Sinkhorn iterations. It also introduces an
entropic bias, which the Sinkhorn divergence removes by subtracting self-costs
[[8]](#ref-feydy):

$$
S_\varepsilon(\rho,\nu)
=
\mathrm{OT}_\varepsilon(\rho,\nu)
-
\frac{1}{2}\mathrm{OT}_\varepsilon(\rho,\rho)
-
\frac{1}{2}\mathrm{OT}_\varepsilon(\nu,\nu).
$$

W-Flow chooses

$$
\mathcal{E}(\rho) = S_\varepsilon(\rho,\rho_{\mathrm{data}})
$$

and follows the Wasserstein gradient flow

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

The corresponding velocity is

$$
v_t(x)
=
-
\nabla
\frac{\delta S_\varepsilon(\rho_t,\rho_{\mathrm{data}})}{\delta \rho}(x).
$$

The procedure has two levels:

1. Define a distributional path from the reference distribution to the data
   distribution using this Wasserstein gradient flow.
2. Train a static generator to approximate the endpoint of that path.

W-Flow is related to drifting through training-time distributional transport.
Its defining features are a fixed energy, an optimal-transport interpretation
of the induced velocity, and an analysis of the finite-sample dynamics against
the continuous-time distributional dynamics [[9]](#ref-han).

## 7. Relation to nearby model classes

| Framework | Main object | Training target | Sampling |
|---|---|---|---|
| Diffusion / score model [[1]](#ref-sohl), [[2]](#ref-song) | Reverse-time SDE or probability-flow ODE | Score $\nabla \log \rho_t$ | Iterative |
| Schrödinger bridge [[3]](#ref-debortoli) | Entropy-regularized path measure | Bridge dynamics | Usually iterative |
| Flow matching [[4]](#ref-lipman) | Velocity field between distributions | Conditional vector field | ODE integration, unless distilled |
| GAN [[10]](#ref-goodfellow) | Static pushforward map | Adversarial divergence proxy | One step |
| Drifting [[5]](#ref-deng) | Training-time pushforward evolution | Drifting field / fixed point | One step |
| W-Flow [[9]](#ref-han) | Wasserstein gradient flow | Sinkhorn-divergence energy descent | One step |

The coarse table keeps one distinction visible.
Diffusion and flow matching typically keep a learned dynamics at sampling time.
GANs, drifting models, and W-Flow aim for one-step generation. Drifting and
W-Flow are worth looking at because they keep a distributional-transport
interpretation and still end with a static generator.

## 8. Compact derivation

The derivation can be summarized as follows.

First, a generator defines a distribution:

$$
\rho_\theta = (G_\theta)_\# \rho_0.
$$

Second, choose an energy that compares the generated distribution with the data
distribution:

$$
\mathcal{E}(\rho) = D(\rho,\rho_{\mathrm{data}}).
$$

For W-Flow, this is

$$
D(\rho,\rho_{\mathrm{data}})
=
S_\varepsilon(\rho,\rho_{\mathrm{data}}).
$$

Third, move the distribution by Wasserstein steepest descent:

$$
\partial_t \rho_t
=
\nabla \cdot
\left(
  \rho_t \nabla \frac{\delta \mathcal{E}}{\delta \rho}
\right).
$$

Equivalently, move particles by

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

Finally, train the generator to absorb this motion:

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

Beyond sampling speed, one-step generation can be connected to a controlled
evolution of probability measures.

This gives a different way to think about the old tension between GAN-like
generators and diffusion-like dynamics. A static generator is fast and requires
a good training signal over distributions. Diffusion gives a strong
distributional signal at the price of repeated inference-time evaluations.
Drifting and W-Flow keep the distributional signal and move the computation into
training.

The main open questions are about the geometry and the finite-sample estimates:
Which energy gives the right motion? How stable is the induced velocity when it
is estimated from batches? How expressive must $G_\theta$ be to absorb the
flow? And when does the fixed point of the training dynamics actually coincide
with the data distribution?

In short: modern one-step generative modeling is starting to look like
amortized distributional transport. Learn the flow during training, then store
the result in a single generator evaluation.

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
