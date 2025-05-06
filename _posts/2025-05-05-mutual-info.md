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