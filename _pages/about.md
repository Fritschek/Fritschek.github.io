---
layout: home
permalink: /
title: "About me"
excerpt: "About me"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

<p align="center">
  <img src="/images/about.jpg" alt="Photo of Rick Fritschek" style="width: 350px;"/>
</p>

I am a research scientist/postdoc at the [Chair of Information Theory and Machine Learning](https://tu-dresden.de/ing/elektrotechnik/ifn/itml/die-professur/inhaber?set_language=en) at [Technische Universität Dresden](https://tu-dresden.de/).
My research studies information flow in neural and communication systems: how information is represented, transmitted, hidden, estimated, and recovered under structural, statistical, computational, and adversarial constraints.

This perspective grew out of my work on communication systems, including interference networks [[1]](/publications/#cellular-deterministic-duality), wiretap channels [[2]](/publications/#gaussian-mac-wiretap-helper), mutual-information estimation [[3]](/publications/#channel-coding-mi-estimation), neural channel coding [[4]](/publications/#mingru-turbo-autoencoder), and generative channel models [[5]](/publications/#diffusion-channel-coding). I am now extending these ideas to learned systems, where information can flow through representations, optimization dynamics, and learned stochastic mechanisms.

I did my Dr.-Ing. (PhD) at Technische Universität Berlin, advised by [Gerhard Wunder](https://scholar.google.de/citations?user=I9ifRZEAAAAJ&hl=de). My thesis was about deterministic models for capacity approximations in interference networks and physical layer security. I received the M.Sc. degree in electrical engineering from [Technische Universität Berlin](https://www.tu.berlin/) in 2012 and the B.Sc. degree in electrical engineering from [Hochschule Furtwangen University](https://www.hs-furtwangen.de/en/) in 2010.

[Email](mailto:rick.fritschek@tu-dresden.de) / [Google Scholar](https://scholar.google.com/citations?user=EfwPnJQAAAAJ&hl=en) / [GitHub](https://github.com/Fritschek) / [LinkedIn](https://de.linkedin.com/in/rickfritschek) / [ORCID](https://orcid.org/0000-0002-2485-5500)

## Research Narrative

The common question behind my work is how information is structured, approximated, hidden, estimated, transmitted, and recovered under constraints. In my earlier work, these constraints were physical, algebraic, or communication-theoretic: interference, secrecy requirements, unknown channels, coding structure, and limited computational resources. This led me to work on channel coding [[3]](/publications/#channel-coding-mi-estimation), wiretap coding [[6]](/publications/#wiretap-coding-mi), interference networks [[1]](/publications/#cellular-deterministic-duality), and mutual-information estimation [[7]](/publications/#neural-mi-estimation).

The same viewpoint extends naturally to learned systems. Neural networks induce implicit information-processing mechanisms through their representations, optimization paths, and model outputs. Understanding these mechanisms is important for privacy, security, robustness, and interpretability: what information is represented, what is discarded, what is hidden, what leaks, and what can be recovered from limited observations?

Communication systems provide a precise foundation for studying learned information processing. In channel coding, information must survive noise. In wiretap coding, information must be hidden from an adversary. In neural channel coding, robust and recoverable representations are learned under strict noise, latency, and compute constraints. The same conceptual tensions reappear in modern learned systems, where the channels are often implicit rather than explicitly specified.

{% comment %}
### Current Direction: Side Channels and Representation Transfer

I currently study how information can be transferred through neural systems even when it is not explicitly represented in the semantic content of the training data. I view subliminal learning as a side-channel transfer problem: a student observes teacher responses on probe inputs that may appear unrelated to a target task, but the responses can still constrain the student in task-relevant directions. Whether transfer occurs depends on probe coverage, non-identifiability of output-only imitation, optimization bias, initialization, and decoding complexity.

This connects directly to my earlier work on secrecy and leakage in communication channels [[2]](/publications/#gaussian-mac-wiretap-helper) [[6]](/publications/#wiretap-coding-mi). The difference is that the channel is no longer only a physical or statistical communication medium. It can be induced by representation geometry, distillation data, model outputs, or the training process itself.
{% endcomment %}

### Research Themes

* **Information-theoretic structure and approximation**: I study how complex systems can be replaced by structured approximations that preserve the relevant information flow. In communication networks, this appears in deterministic models, interference geometry, capacity approximations, and duality relations [[1]](/publications/#cellular-deterministic-duality) [[8]](/publications/#gaussian-imac-constant-gap). In learned systems, the analogous question is which task-relevant, private, or hidden information survives abstraction, compression, and optimization.
* **Security, privacy, and hidden channels**: My work on wiretap coding and physical-layer security studied how to communicate reliably while limiting adversarial inference [[2]](/publications/#gaussian-mac-wiretap-helper) [[6]](/publications/#wiretap-coding-mi). I am now extending this view to neural systems, where leakage can occur through outputs, representations, gradients, or training protocols.
* **Estimating information in learned systems**: Mutual information is central to communication theory, but it becomes difficult to estimate in high-dimensional learned systems. I study neural mutual-information estimation, estimator behavior, and information diagnostics for learned representations [[7]](/publications/#neural-mi-estimation) [[9]](/publications/#reverse-jensen-mi). This line of work supports the broader question of how to measure what is preserved or leaked in implicit channels.
* **Learned channels and neural communication**: Neural communication systems provide a controlled setting for studying representation, recovery, memory, and generalization under noise and compute constraints. My recent work uses diffusion models as learned channel approximations and recurrent architectures as scalable neural coding mechanisms [[4]](/publications/#mingru-turbo-autoencoder) [[5]](/publications/#diffusion-channel-coding) [[10]](/publications/#diffusion-channel-distributions).

## Selected Projects and Code

{% for project in site.data.projects %}
* **[{{ project.title }}]({{ project.url }})**: {{ project.description }}{% if project.links %}{% for link in project.links %} [[{{ link.label }}]({{ link.url }})]{% endfor %}{% endif %}
{% endfor %}

## Recent News

* February 2026. Our collaborative paper "AI/ML-Driven 6G Network Solutions with Energy Efficiency Considerations" appeared in *IEEE Access*.
* January 2026. Our collaborative paper "6G PHY: Insights From 6G-ANNA Research Initiative" appeared in *IEEE Open Journal of the Communications Society*.
{% comment %}
* 2026. I am working on side channels, subliminal learning, and representation transfer in learned systems.
{% endcomment %}
* May 2025. My paper "MinGRU-Based Encoder for Turbo Autoencoder Frameworks" with [Rafael Schaefer](https://scholar.google.de/citations?user=PrTUgYQAAAAJ&hl=de) appeared at ICMLCN 2025.
* May 2024. My colleague [Muah Kim](https://sites.google.com/view/muahkim) gave a [tutorial](https://github.com/Fritschek/MinDiffusion/blob/main/Slides.pdf) about diffusion models at ICMLCN 2024 based on our work.
* January 2023. Our paper "Learning End-to-End Channel Coding with Diffusion Models" with [Muah Kim](https://sites.google.com/view/muahkim) and [Rafael Schaefer](https://scholar.google.de/citations?user=PrTUgYQAAAAJ&hl=de) was accepted.

## Contact

Email: rick.fritschek at tu-dresden.de, rickfritschek at gmail.com

<address>
Technische Universität Dresden<br />
Chair of Information Theory and Machine Learning<br />
01062 Dresden, Germany
</address>
