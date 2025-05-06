---
layout: single
title: "Why Any Distribution-Free Lower-Bound Estimator for Mutual Information Can’t Beat ln N"
date: 2025-05-05
tags: [mutual-information, estimation, impossibility]
---

> *An explainer of McAllester & Stratos (2020) plus worked intuition,
> aimed at anyone puzzled by the “$\ln N$ saturation’’ of neural MI estimators.*

---

## 1  Background & Motivation

Modern representation-learning methods often maximise or estimate
mutual information (MI):

* **InfoMax** (Bell & Sejnowski 1995)  
* **Contrastive Predictive Coding** (Oord et al. 2018)  
* **MINE** (Belghazi et al. 2018)  
* InfoNCE / SimCLR variants  

All rely on a sample-based lower bounds $\widehat I_{\mathrm{LB}} \le I(X;Y)$
computed on a minibatch or sample of size $N$.
Empirically one sees a ceiling

$$
\widehat I_{\mathrm{LB}} \;\approx\; \ln N
\quad\text{(nats)}
\quad\bigl(\log_2 N \text{ bits}\bigr).
$$

McAllester & Stratos (2020) prove this ceiling is
**information-theoretically unavoidable** if you demand

1. **Distribution-free:** works for *every* underlying distribution.  
2. **High confidence:**  
   $\Pr\bigl[\widehat B(S) \le \text{true value}\bigr] \ge 1-\delta$.

---

## 2  The Limitation in One Line

**Theorem (informal).**  
For any such estimator using $N$ samples,

$$
\boxed{\;
  \widehat B(S)\; \le\; C\,\ln N
\;}
\qquad(\text{with high probability}),
$$

whether $\widehat B$ targets KL divergence, entropy, or mutual information.

---

## 3  Geometric Intuition — The Hidden Spike

1. Start with a benign distribution $p$.  
2. Adversary forms  

   $$
   \tilde p(x) \;=\;
   \Bigl(1-\tfrac1N\Bigr)\,p(x)
   \;+\;
   \tfrac1N\,s(x),
   $$

   where $s(x)$ is a razor-thin **spike** (or unseen symbols in discrete
   space).  
3. With probability $(1-\tfrac1N)^N \gtrsim e^{-1}$ **no sample** lands
   in the spike, so the batch looks identical to one from $p$.  
4. Yet the spike can reduce the true entropy/KL/MI to
   $\le \ln N$.  
5. Any lower-bound routine certifying $>\ln N$ would be wrong on that
   batch $\Rightarrow$ must never exceed $O(\ln N)$.

---


The picture should show:

* orange — baseline uniform pdf $p(x)$;  
* red — spike holding $1/N$ mass in width $w$;  
* black — $N$ samples, none in spike.

See the code in §8 to generate the figure.

---

## 4  Formal Sketch (KL Version)

1. **Goal**  Output $B(S)$ with  
   $B(S) \le D_{\mathrm{KL}}(p\Vert q)$ (confidence $1-\delta$).  

2. **Adversary**  Mix $q$ with $p$:

  $$
   \tilde q(x)=\Bigl(1-\tfrac1N\Bigr)q(x)+\tfrac1N\,p(x).
   $$

3. **True KL is small**  
   $\tilde q(x)\ge \tfrac1N p(x)\;\Rightarrow\;
    D_{\mathrm{KL}}(p\Vert\tilde q)\le\ln N.$

4. **Indistinguishability**  
   No-spike sample $\;\Rightarrow\; S\sim \tilde q^N$
   indistinguishable from $q^N$.

5. **Therefore**  $B(S) > \ln N + \mathrm{const}$ would violate the
   guarantee with $\ge 25\%$ probability.  
   Hence $B(S) \le \ln N + \mathrm{const}$ w.h.p.

A similar construction yields the entropy bound; since
$I(X;Y)$ is a KL, the same ceiling applies to MI.

---

## 5  Why the Ceiling Is Logarithmic

A hidden mass $\tfrac1N$ has information content
$-\ln\tfrac1N = \ln N$.  
That is the **maximum surprise** a never-observed event can carry;
any universal bound must hedge against it.

---

## 6  Upper Bounds Are Fine

Missing a rare event *under-estimates* entropy/KL, which is harmless for
**upper** bounds.  
Simple concentration gives

$$
H(p)\;\le\; \widehat H_{\text{emp}}
          + O\!\Bigl(\sqrt{\tfrac{\ln(1/\delta)}{N}}\Bigr),
$$

no $\ln N$ obstruction.  
This motivates *Difference-of-Entropies* estimation:

$$
I(X;Y) = H(X) - H(X\!\mid Y)
       \approx
       \widehat H_{\text{cross}}(X)
       -\widehat H_{\text{cross}}(X\!\mid Y),
$$

which lacks a formal lower-bound guarantee but measures large MI in
practice.

---

## 7  Practical Lessons

* Without strong assumptions (finite alphabet, parametric family,
  smoothness), **large certified lower bounds are impossible**.  
* Prefer estimation strategies that do **not** rely on universal lower
  bounds — e.g. cross-entropy upper bounds or downstream proxy tasks.  

---

## 8  Python Simulation Template
```python
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
