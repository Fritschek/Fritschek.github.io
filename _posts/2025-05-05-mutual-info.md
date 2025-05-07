---
layout: single
title: "Why Any Distribution-Free Lower-Bound Estimator for Mutual Information Can’t Beat ln N"
date: 2025-05-05
tags: [mutual-information, estimation, impossibility]
---

> *An explainer of McAllester & Stratos (2020) plus worked intuition,
> aimed at anyone puzzled by the “$\ln N$ saturation’’ of neural MI estimators.*

---

## 1. Background & Motivation

Mutual information (MI) plays a central role in information theory, communication theory and modern representation learning. Recent advances in neural networks, and the easier handling of variational results for practical application led to a rejuvenation of old representations of Kullback-Leibler divergence, i.e., the Donsker-Varadhan based lower bound, which was investigated in MINE. Also other lower bounds for mutual information got renewed interest such as the Nguyien-Wainwright-Jordan lower bound. Or the more recent noise-contrastive method, the InfoNCE. In representation learning, these estimators can be used to learn from maximizing feature information in data. In communication theory, these estimators can be used to learn for example optimal encoding. I refer to the overview paper by Poole et al for a comprehensive overview.

All these recent bounds rely on lower bounds of mutual information estimated from finite samples. Formally, given a sample of size $N$, the algorithm computes a bound $\widehat I_{\mathrm{LB}} \le I(X;Y)$, hoping that it gets close to the true value, be approximating from below.

However, in practice the bounds MINE, NWJ and derivtes exibit high variance, and estimates fluctuate below AND above the true MI value, seemingly contradicting the theoretical results. The InfoNCE bound exibits very low variance but its MI value is limited to $\log N$, where $N$ is the batch size.

McAllester and Stratos (2020) showed that this behavior is an inherent **limitation**. If an estimator is required to work on arbitrary distributions (i.e., “distribution-free”) and to provide valid lower bounds with high probability (say, with confidence $1 - \delta$), then it cannot exceed a constant times $\log N$. In other words, **no universal, high-confidence lower bound can grow faster than logarithmically in the sample size**.

---

## 2. Intuition — (Hidden Spikes in Data)

### 2.1 The discrete case
Lets have a look at a uniform distribution, which maximizes the entropy.
We know that $I(X;Y) = H(X) - H(X;Y) \le H(X)$. I.e. MI is a lower bound for entropy. Now, a uniform distribution on some finite interval maximizes the entropy. Any spike in this distribution lowers the entropy. So the sampling mechanism needs to hit the spike, to accurately estimate the entropy. As the entropy upper bounds the MI, it can be seen how this problem directly translates to MI.

### 2.2 The continues case

To see why the $\log N$ ceiling is unavoidable, consider a simple trick an adversary can play on your data.

Start with a nice, well-behaved distribution $p(x)$. Now define a new distribution $\tilde{p}(x)$ that’s almost identical to $p$, except it hides a tiny spike $s(x)$:

$$
\tilde{p}(x) = \left(1 - \frac{1}{N} \right) p(x) + \frac{1}{N} s(x),
$$

where $s(x)$ is sharply concentrated on a narrow region or unseen symbol. This spike carries just $1/N$ of the total probability mass.

Now sample $N$ points from $\tilde{p}$. With probability $(1-\frac{1}{N})^N$, the sample never hits the spike so the sample is indistinguishable from one drawn from $p$. For $N=2$ this is $1/4$, converging to $e^{-1}$ for $N\leftarrow \infty$.  Yet the spike can drastically lower the entropy, KL divergence, or mutual information of the true distribution, as argued above.

### 2.3 Sketch of the $\log N$ bound (KL Version)

Let’s see how this limitation plays out in the case of KL divergence.

Suppose you want to estimate $D_{\mathrm{KL}}(p \Vert q)$ from a finite sample $S \sim p^N$, and your estimator $B(S)$ is required to be a high-confidence lower bound. That is, it must satisfy

$$
\Pr\left[ B(S) \le D_{\mathrm{KL}}(p \Vert q) \right] \ge 1 - \delta.
$$

Now imagine the adversary modifies $q$ slightly by mixing in a bit of $p$, creating a new distribution:

$$
\tilde{q}(x) = \left(1 - \frac{1}{N} \right) q(x) + \frac{1}{N} p(x).
$$

This change guarantees that $\tilde{q}(x) \ge \frac{1}{N} p(x)$, and from this, it follows that

$$
D_{\mathrm{KL}}(p \Vert \tilde{q}) \le \ln N.
$$

At the same time, a batch of $N$ samples from $p$ is statistically very unlikely to detect the difference between $q$ and $\tilde{q}$, since the mass added to $p$ is only $1/N$. In fact, samples from $\tilde{q}^N$ and $q^N$ are nearly indistinguishable unless one of them lands in the spike — which happens with low probability.

So if the estimator ever outputs a value greater than $\ln N$ on a batch that looks like it came from $q$, it risks being wrong under $\tilde{q}$ with nontrivial probability — violating the confidence guarantee.

The safest strategy for the estimator is to stay below $\ln N + \text{const}$, regardless of the true KL. And since mutual information is itself a KL divergence, this same limitation applies directly to MI lower bounds as well.

---

## 5. Why the Ceiling Is Logarithmic

The $\ln N$ ceiling isn’t arbitrary — it has a clear information-theoretic origin. If an event has probability $1/N$, its information content is $-\ln(1/N) = \ln N$. That’s the amount of “surprise” you’d experience upon seeing it.

Now consider a spike with mass $1/N$ that goes unobserved in a batch of size $N$. To guard against the possibility that such a spike exists — and that it dramatically reduces the true entropy or KL divergence — any estimator must be conservative. It can’t claim more than $\ln N$ without risking overestimation in adversarial cases.

This is the key point: the logarithmic growth reflects the maximum information an unseen but plausible event could carry. No estimator that promises universal, high-confidence lower bounds can afford to ignore that risk.

---

## 6. Upper Bounds Are Fine

Interestingly, this limitation only applies to **lower bounds**. If you're trying to *underestimate* something like entropy or mutual information, missing rare events is dangerous. But if you're estimating from above — say, bounding entropy from above — then missing a spike is harmless.

This is because any rare, unobserved event can only make the true value lower than your estimate. As a result, simple concentration arguments can give you high-confidence **upper bounds** on entropy like:

$$
H(p) \le \widehat{H}_{\text{emp}} + O\left( \sqrt{\frac{\ln(1/\delta)}{N}} \right),
$$

where $\widehat{H}_{\text{emp}}$ is the empirical entropy from the sample.

This logic underlies practical estimators like the **difference-of-entropies** approach to mutual information:

$$
I(X;Y) = H(X) - H(X \mid Y) \approx \widehat{H}_{\text{cross}}(X) - \widehat{H}_{\text{cross}}(X \mid Y),
$$

which uses cross-entropy losses as proxies for upper bounds. While these do not guarantee a lower bound on $I(X;Y)$, they’re often able to track large mutual information values effectively in practice — especially when paired with good models and sufficient data.


## 7  Practical Lessons

* Without strong assumptions (finite alphabet, parametric family,
  smoothness), **large certified lower bounds are impossible**.  
* Prefer estimation strategies that do **not** rely on universal lower
  bounds — e.g. cross-entropy upper bounds or downstream proxy tasks.  

---

## 8

<details>
<summary>Show Python code</summary>

{% highlight python %} 
import numpy as np
import matplotlib.pyplot as plt

N = 100          # batch size
eps = 1/N        # hidden mass
width = 0.003    # spike width
pos = 0.8        # spike start

# pdfs on [0,1]
x = np.linspace(0, 1, 2000)
p_pdf = np.ones_like(x)
q_pdf = np.ones_like(x)*(1-eps)
q_pdf += ((x>=pos) & (x<=pos+width)) * (eps/width)

# sample N points from q_pdf  (naive rejection sampling)
rng = np.random.default_rng(0)
samps = []
while len(samps) < N:
    u = rng.random()
    if rng.random() <= q_pdf[(np.abs(x-u)<1e-3)][0] / q_pdf.max():
        samps.append(u)
samps = np.array(samps)

# plot
plt.figure(figsize=(9,3))
plt.plot(x, p_pdf, label='p: uniform')
plt.plot(x, q_pdf, label='tilde p: uniform + spike')
plt.axvspan(pos, pos+width, color='red', alpha=0.25, label='spike')
plt.scatter(samps, np.zeros_like(samps), marker='|', s=80, color='k', label='samples')
plt.ylim(0, q_pdf.max()*1.1)
plt.xlabel('x'); plt.ylabel('pdf')
plt.title(f'Adversarial spike (ε={eps}, unseen by N={N} samples)')
plt.legend(frameon=False); plt.tight_layout()
plt.savefig('adversarial_spike.png', dpi=150)
```
{% endhighlight %} 

<\details>
