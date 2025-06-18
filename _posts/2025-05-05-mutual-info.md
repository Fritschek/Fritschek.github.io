---
layout: single
title: "Why Any Distribution-Free Lower-Bound Estimator for Mutual Information Can’t Beat ln N"
date: 2025-05-05
tags: [mutual-information, estimation, impossibility]
---

*A short explanation of McAllester & Stratos' (2020) mututal information estimation result plus some intuition.*

---

## 1. Background & Motivation

Mutual information (MI) plays a central role in information theory, communication theory and modern representation learning. 
Recent advances in neural networks, and the easier realisation of variational results for practical applications led to a rejuvenation of old representations of Kullback-Leibler divergence, 
i.e., the Donsker-Varadhan (DV) based lower bound, which was investigated in MINE [^Belghazi18]. 

Also other lower bounds for mutual information got renewed interest such as the Nguyien-Wainwright-Jordan (NWJ) lower bound [^NWJ10]. 
Or the more recent noise-contrastive method, the InfoNCE [^NCE10]. In representation learning, these estimators can be used to learn from maximizing feature information in data. 
In communication theory, these estimators can be used to learn for example optimal encoding [^Fritschek19]. I refer to the overview paper by Poole et al [^Poole19] for a comprehensive overview.

All these recent bounds rely on lower bounds of mutual information estimated from finite samples. 
Formally, given a sample of size $N$, the algorithm computes a bound $\widehat I_{\mathrm{LB}} \le I(X;Y)$, hoping that it gets close to the true value, by approximating from below.

However, in practice the bounds MINE, NWJ and others exibit high variance, and estimates fluctuate below AND above the true MI value, seemingly contradicting the theoretical results. The InfoNCE bound exibits very low variance but its MI value is limited to $\log N$, where $N$ is the batch size.

[Note that Poole et al already showed that using Monte-Carlo approximation of the expectation terms in MINE, i.e. 
$\mathcal{L}_{\text{MINE}}= \mathbb{E}_{p_{XY}}[f_\theta(X,Y)]- \log \mathbb{E}_{p_X p_Y}\!\bigl[e^{f_\theta(X,Y)}\bigr]$, yields neither lower nor upper bound due to the nonlinearity(log).]

McAllester and Stratos (2020) showed that this behavior is an inherent **limitation**. 
If an estimator is required to work on arbitrary distributions (i.e., “distribution-free”) and to provide valid lower bounds with high probability (say, with confidence $1 - \delta$), 
then it cannot exceed a constant times $\log N$. In other words, **no universal, high-confidence lower bound can grow faster than logarithmically in the sample size**.


---

## 2. Intuition — (Hidden Spikes in Data)

### 2.1 The discrete case
Lets have a look at a uniform distribution, which maximizes the entropy.
We know that $I(X;Y) = H(X) - H(X;Y) \le H(X)$. So MI is a lower bound for entropy. Now, a uniform distribution on some finite interval maximizes the entropy. 
Any spike in this distribution lowers the entropy. So the sampling mechanism needs to hit the spike, to accurately estimate the entropy. 
As the entropy upper bounds the MI, it can be seen how this problem directly translates to MI.

### 2.2 The continues case

To see why the $\log N$ ceiling is unavoidable, consider a simple trick an adversary can play on your data.

Start with a nice, well-behaved distribution $p(x)$. Now define a new distribution $\tilde{p}(x)$ that’s almost identical to $p$, except it hides a tiny spike $s(x)$:

$$
\tilde{p}(x) = \left(1 - \frac{1}{N} \right) p(x) + \frac{1}{N} s(x),
$$

where $s(x)$ is sharply concentrated on a narrow region. This spike carries just $1/N$ of the total probability mass.

Now sample $N$ points from $\tilde{p}$. With probability $(1-\frac{1}{N})^N$, the sample never hits the spike so the sample is indistinguishable from one drawn from $p$. 
For $N=2$ this is $1/4$, converging to $e^{-1}$ for $N\leftarrow \infty$.  Yet the spike can drastically lower the entropy, KL divergence, or mutual information of the true distribution, as argued above.

### 2.3 Sketch of the $\log N$ bound (KL Version)

Let’s see how this limitation plays out in the case of KL divergence.

Suppose you want to estimate $D_{\mathrm{KL}}(p \Vert q)$ from a finite sample $S \sim p^N$, and your estimator $E(S)$ is required to be a high-confidence lower bound. 
That is, it must satisfy

$$
\Pr\left[ E(S) \le D_{\mathrm{KL}}(p \Vert q) \right] \ge 1 - \delta.
$$

Now imagine we have the same adversary as above, which modifies $q$ slightly by mixing in a bit of $p$, creating a new distribution:

$$
\tilde{q}(x) = \left(1 - \frac{1}{N} \right) q(x) + \frac{1}{N} p(x).
$$

This change guarantees that $\tilde{q}(x) \ge \frac{1}{N} p(x)$, and from this, it follows that

$\frac{\tilde{q}(x)}{p(x)}\le N$ and therefore $\mathbb{E}[\log \frac{\tilde{q}(x)}{p(x)}]\le \log N$ which is

$$
D_{\mathrm{KL}}(p \Vert \tilde{q}) \le \log N.
$$

At the same time, a batch of $N$ samples from $p$ is statistically very unlikely to detect the difference between $q$ and $\tilde{q}$, since the mass added to $p$ is only $1/N$.
In fact, samples from $\tilde{q}^N$ and $q^N$ are nearly indistinguishable unless one of them lands in the spike, which happens with low probability.
In fact, as argued above, the chance for a switch up is greater than $1/4$.

So if the estimator ever outputs a value greater than $\log N$ on a batch that looks like it came from $q$, it risks being wrong under $\tilde{q}$ with nontrivial probability ($e^{-1}$ in the limit) which violating the confidence guarantee.

The safest strategy for the estimator is to stay below $\log N + \text{const}$, regardless of the true KL unless it has specific structural properties. 
And since mutual information is itself a KL divergence, this same limitation applies directly to MI lower bounds as well.

---
## 7  Lessons

* Without strong assumptions (finite alphabet, parametric family,
  smoothness), **large lower bounds are impossible**.

## Refs

[^Belghazi18]: Belghazi, M. I. *et al.* (2018).  
  **Mutual Information Neural Estimation (MINE)**.  
  *Proceedings of the 35th International Conference on Machine Learning*.
[^NWJ10]: Nguyen, X., Wainwright, M. J., & Jordan, M. I. (2010).  
**Estimating Divergence Functionals and the Likelihood Ratio by Convex Risk Minimization.**  
*IEEE Transactions on Information Theory*, 56 (11), 5847-5861.  

[^NCE10]: Gutmann, M., & Hyvärinen, A. (2010).  
**Noise-Contrastive Estimation: A New Estimation Principle for Unnormalized Statistical Models.**  
In *Proceedings of AISTATS 13*, 297-304.  

[^Fritschek19]: Fritschek, R., Schaefer, R. F., & Wunder, G. (2019).  
**Deep Learning for Channel Coding via Neural Mutual Information Estimation.**  
In *Proceedings of the 2019 IEEE 20th International Workshop on Signal Processing Advances in Wireless Communications* (SPAWC), 1-5.  

[^Poole19]: Poole, B., Ozair, S., van den Oord, A., Alemi, A. A., & Tucker, G. (2019).  
&nbsp;&nbsp;**On Variational Bounds of Mutual Information.**  
In *Proceedings of the 36th International Conference on Machine Learning* (ICML), 5171-5180.

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
