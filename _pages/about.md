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

I am a research scientist/postdoc at the [Chair of Information Theory and Machine Learning](https://tu-dresden.de/ing/elektrotechnik/ifn/itml/die-professur/inhaber?set_language=en) at [Technische Universität Dresden](https://tu-dresden.de/) where I work at the intersection of information theory and machine learning.
I did my Dr.-Ing. (PhD) at Technische Universität Berlin, advised by [Gerhard Wunder](https://scholar.google.de/citations?user=I9ifRZEAAAAJ&hl=de). My thesis was about deterministic models for capacity approximations in interference networks and physical layer security. I received the M.Sc. degree in electrical engineering from [Technische Universität Berlin](https://www.tu.berlin/) in 2012 and the B.Sc. degree in electrical engineering from [Hochschule Furtwangen University](https://www.hs-furtwangen.de/en/) in 2010.

[Email](mailto:rick.fritschek@tu-dresden.de) / [Google Scholar](https://scholar.google.com/citations?user=EfwPnJQAAAAJ&hl=en) / [GitHub](https://github.com/Fritschek) / [LinkedIn](https://de.linkedin.com/in/rickfritschek) / [ORCID](https://orcid.org/0000-0002-2485-5500)

## Research

I am generally interested in information theory, security, fairness, machine learning, and statistics. My research spans the following topics:

* Information theory: theoretical limits of data compression and transmission
* High-dimensional statistics: estimation methods and theoretical properties in high-dimensional spaces, particularly for applications in AI
* AI for wireless communication: diffusion models and the interplay between classical codes and neural coding schemes
* Mutual information estimation and its applications in communication systems

## Selected Projects and Code

{% for project in site.data.projects %}
* **[{{ project.title }}]({{ project.url }})**: {{ project.description }}{% if project.links %}{% for link in project.links %} [[{{ link.label }}]({{ link.url }})]{% endfor %}{% endif %}
{% endfor %}

## Recent News

* February 2026. Our collaborative paper "AI/ML-Driven 6G Network Solutions with Energy Efficiency Considerations" appeared in *IEEE Access*.
* January 2026. Our collaborative paper "6G PHY: Insights From 6G-ANNA Research Initiative" appeared in *IEEE Open Journal of the Communications Society*.
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
