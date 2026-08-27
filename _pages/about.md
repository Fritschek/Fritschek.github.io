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
  <img src="/images/about.jpg" alt="Photo of Rick Fritschek" style="width: 350px"/>
</p>

I am a research scientist/postdoc at the [Chair of Information Theory and Machine Learning](https://tu-dresden.de/ing/elektrotechnik/ifn/itml/die-professur/inhaber?set_language=en) at [Technische Universität Dresden](https://tu-dresden.de/).
My research studies information flow in neural and communication systems: which parts are represented, which parts are protected, and what can be recovered under structural, statistical, computational, or adversarial constraints.

This perspective grew out of work on communication systems, including interference networks [[1]](/publications/#cellular-deterministic-duality), wiretap channels [[2]](/publications/#gaussian-mac-wiretap-helper), mutual-information estimation [[3]](/publications/#channel-coding-mi-estimation), neural channel coding [[4]](/publications/#mingru-turbo-autoencoder), and generative channel models [[5]](/publications/#diffusion-channel-coding). The current focus is on learned systems, where information flows through representations, optimization dynamics, and learned stochastic mechanisms.

I did my Dr.-Ing. (PhD) at Technische Universität Berlin, advised by [Gerhard Wunder](https://scholar.google.de/citations?user=I9ifRZEAAAAJ&hl=de). My thesis was about deterministic models for capacity approximations in interference networks and physical layer security. I received the M.Sc. degree in electrical engineering from [Technische Universität Berlin](https://www.tu.berlin/) in 2012 and the B.Sc. degree in electrical engineering from [Hochschule Furtwangen University](https://www.hs-furtwangen.de/en/) in 2010.

[Email](mailto:rick.fritschek@tu-dresden.de) / [Google Scholar](https://scholar.google.com/citations?user=EfwPnJQAAAAJ&hl=en) / [GitHub](https://github.com/Fritschek) / [LinkedIn](https://de.linkedin.com/in/rickfritschek) / [ORCID](https://orcid.org/0000-0002-2485-5500)

## Research

The research problem is information flow under constraints: how structure changes what can be transmitted, hidden, estimated, or recovered. In my earlier work, these constraints were physical, algebraic, or communication-theoretic: interference, secrecy requirements, unknown channels, coding structure, and limited computational resources. This line includes channel coding [[3]](/publications/#channel-coding-mi-estimation), wiretap coding [[6]](/publications/#wiretap-coding-mi), interference networks [[1]](/publications/#cellular-deterministic-duality), and mutual-information estimation [[7]](/publications/#neural-mi-estimation).

In learned systems, neural networks induce implicit information-processing mechanisms through representations, optimization paths, and model outputs. This raises questions about leakage and interpretability: what is represented, what is discarded, what leaks, and what can be recovered from limited observations?

Communication systems provide a precise foundation for studying learned information processing. In channel coding, information must survive noise. In wiretap coding, information must be hidden from an adversary. In neural channel coding, recoverable representations are learned under strict noise, latency, or compute constraints. Modern learned systems face related questions when channels arise implicitly from architectures, objectives, or data.

<div class="research-map" aria-labelledby="research-map-title">
  <h3 id="research-map-title">Research Map</h3>

  <div class="research-map__flow" role="img" aria-label="Information flow under constraints connects communication systems with learned systems.">
    <div class="research-map__node research-map__node--source">
      <span class="research-map__eyebrow">Earlier setting</span>
      <strong>Communication systems</strong>
      <span>interference, secrecy, coding, unknown channel laws</span>
    </div>
    <div class="research-map__connector" aria-hidden="true"></div>
    <div class="research-map__core">
      <span class="research-map__eyebrow">Common object</span>
      <strong>Information flow under constraints</strong>
      <span>represented · transmitted · protected · estimated · recovered</span>
    </div>
    <div class="research-map__connector" aria-hidden="true"></div>
    <div class="research-map__node research-map__node--target">
      <span class="research-map__eyebrow">Learned setting</span>
      <strong>Learned systems</strong>
      <span>representations, optimization paths, generative mechanisms</span>
    </div>
  </div>

  <div class="research-map__themes" aria-label="Research themes with publication links">
    <section class="research-map__theme research-map__theme--structure">
      <h4>Structure and approximation</h4>
      <p>Deterministic models, interference geometry, capacity approximations.</p>
      <p class="research-map__refs"><a href="/publications/#cellular-deterministic-duality">[1]</a> <a href="/publications/#gaussian-imac-constant-gap">[8]</a></p>
    </section>
    <section class="research-map__theme research-map__theme--security">
      <h4>Security and privacy</h4>
      <p>Wiretap channels, secrecy constraints, adversarial inference.</p>
      <p class="research-map__refs"><a href="/publications/#gaussian-mac-wiretap-helper">[2]</a> <a href="/publications/#wiretap-coding-mi">[6]</a></p>
    </section>
    <section class="research-map__theme research-map__theme--estimation">
      <h4>Information estimation</h4>
      <p>Sample-based MI estimation, estimator behavior, information diagnostics.</p>
      <p class="research-map__refs"><a href="/publications/#neural-mi-estimation">[7]</a> <a href="/publications/#reverse-jensen-mi">[9]</a></p>
    </section>
    <section class="research-map__theme research-map__theme--learned">
      <h4>Learned channels</h4>
      <p>Diffusion channel models, neural coding, sequential inductive bias.</p>
      <p class="research-map__refs"><a href="/publications/#mingru-turbo-autoencoder">[4]</a> <a href="/publications/#diffusion-channel-coding">[5]</a> <a href="/publications/#diffusion-channel-distributions">[10]</a></p>
    </section>
  </div>
</div>

{% comment %}
### Current Direction: Side Channels and Representation Transfer

This section is intentionally withheld until the relevant paper is public.
{% endcomment %}

### Research Themes

* **Information-theoretic structure and approximation**: Complex systems can often be replaced by structured approximations that preserve the relevant information flow. In communication networks, this appears in deterministic models, interference geometry, capacity approximations, and duality relations [[1]](/publications/#cellular-deterministic-duality) [[8]](/publications/#gaussian-imac-constant-gap). In learned systems, the analogous question is which task-relevant, private, or hidden information survives abstraction and compression.
* **Security and privacy**: My work on wiretap coding and physical-layer security studied reliable communication under adversarial inference constraints [[2]](/publications/#gaussian-mac-wiretap-helper) [[6]](/publications/#wiretap-coding-mi). The same tools also apply to leakage in neural systems, where adversarial inference can exploit outputs, representations, gradients, or training protocols.
* **Estimating information in learned systems**: Mutual information anchors communication theory, but it becomes difficult to estimate in high-dimensional learned systems. I study neural mutual-information estimation, estimator behavior, and information diagnostics for learned representations [[7]](/publications/#neural-mi-estimation) [[9]](/publications/#reverse-jensen-mi). This line of work supports the broader question of how to measure what is preserved or leaked in implicit channels.
* **Learned channels and neural communication**: Neural communication systems provide a controlled setting for studying representation and recovery under noise and compute constraints. Recent work uses diffusion models as learned channel approximations and recurrent architectures as scalable neural coding mechanisms [[4]](/publications/#mingru-turbo-autoencoder) [[5]](/publications/#diffusion-channel-coding) [[10]](/publications/#diffusion-channel-distributions).

## Selected Projects and Code

{% for project in site.data.projects %}
* **[{{ project.title }}]({{ project.url }})**: {{ project.description }}{% if project.links %}{% for link in project.links %} [[{{ link.label }}]({{ link.url }})]{% endfor %}{% endif %}
{% endfor %}

## Recent News

* August 2026. A GLOBECOM version of "Condition-Wise Sinkhorn Drifting for One-Shot Learned Channel Simulation" was accepted.
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
