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

Mutual information (MI) plays a central role in modern representation learning. From the classic InfoMax principle of Bell & Sejnowski (1995), to more recent self-supervised methods like Contrastive Predictive Coding (CPC), MINE, and InfoNCE-based models such as SimCLR, estimating or maximizing MI has become a popular strategy for learning useful features from data.

Despite the diversity of these methods, they all share a common foundation: they rely on lower bounds of mutual information estimated from finite samples. Formally, given a sample of size $N$, the algorithm computes a bound $\widehat I_{\mathrm{LB}} \le I(X;Y)$, hoping that it gets close to the true value.

However, practitioners have noticed a curious and frustrating pattern: no matter how clever the method, these lower bounds tend to saturate around $\ln N$ nats (or $\log_2 N$ bits). For example, doubling the batch size leads to roughly a $\ln 2$ increase in the bound—but only up to a point. Beyond that, improvements flatten out.

McAllester and Stratos (2020) showed that this behavior isn’t just a practical nuisance—it’s an **information-theoretic limitation**. If your estimator is required to work on arbitrary distributions (i.e., “distribution-free”) and to provide valid lower bounds with high probability (say, with confidence $1 - \delta$), then it cannot exceed a constant times $\ln N$. In other words, **no universal, high-confidence lower bound can grow faster than logarithmically in the sample size**.

---

## 3. Geometric Intuition — The Hidden Spike

To see why the $\ln N$ ceiling is unavoidable, consider a simple trick an adversary can play on your data.

Start with a nice, well-behaved distribution $p(x)$. Now define a new distribution $\tilde{p}(x)$ that’s almost identical to $p$, except it hides a tiny spike:

$$
\tilde{p}(x) = \left(1 - \frac{1}{N} \right) p(x) + \frac{1}{N} s(x),
$$

where $s(x)$ is sharply concentrated on a narrow region or unseen symbol. This spike carries just $1/N$ of the total probability mass.

Now sample $N$ points from $\tilde{p}$. With probability close to $e^{-1}$, none of them land in the spike — so the sample is indistinguishable from one drawn from $p$. Yet the spike can drastically lower the entropy, KL divergence, or mutual information of the true distribution.

If a lower-bound estimator were to output a value larger than $\ln N$, it would be wrong on such a batch with non-negligible probability. To avoid this, it must stay below $O(\ln N)$, even when the true value is higher. That’s the geometric core of the impossibility.

---

## 4. Formal Sketch (KL Version)

Let’s sketch how this limitation plays out in the case of KL divergence.

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
