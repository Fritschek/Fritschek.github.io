---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

An up-to-date list of my articles can be found on <u><a href="https://scholar.google.com/citations?user=EfwPnJQAAAAJ&hl=en">my Google Scholar profile</a></u> and my <u><a href="https://tu-dresden.de/ing/elektrotechnik/ifn/itml/die-professur/team/fritschek?set_language=en">TU Dresden FIS profile</a></u>.

{% include base_path %}

## Journal Articles

{% for pub in site.data.publications.journal %}
{{ forloop.index }}. {{ pub.authors }}, "{{ pub.title }}"{% if pub.venue %}, in *{{ pub.venue }}*{% endif %}{% if pub.details %}, {{ pub.details }}{% endif %}.{% if pub.links %}{% for link in pub.links %} [[{{ link.label }}]({{ link.url }})]{% endfor %}{% endif %}

{% endfor %}

## Conference Articles

{% for pub in site.data.publications.conference %}
{{ forloop.index }}. {{ pub.authors }}, "{{ pub.title }}"{% if pub.venue %}, in *{{ pub.venue }}*{% endif %}{% if pub.details %}, {{ pub.details }}{% endif %}.{% if pub.links %}{% for link in pub.links %} [[{{ link.label }}]({{ link.url }})]{% endfor %}{% endif %}

{% endfor %}

## Book Chapters

{% for pub in site.data.publications.book_chapters %}
{{ forloop.index }}. {{ pub.authors }}, "{{ pub.title }}"{% if pub.venue %}, in *{{ pub.venue }}*{% endif %}{% if pub.details %}, {{ pub.details }}{% endif %}.{% if pub.links %}{% for link in pub.links %} [[{{ link.label }}]({{ link.url }})]{% endfor %}{% endif %}

{% endfor %}

## Preprints

{% for pub in site.data.publications.preprints %}
{{ forloop.index }}. {{ pub.authors }}, "{{ pub.title }}"{% if pub.venue %}, in *{{ pub.venue }}*{% endif %}{% if pub.details %}, {{ pub.details }}{% endif %}.{% if pub.links %}{% for link in pub.links %} [[{{ link.label }}]({{ link.url }})]{% endfor %}{% endif %}

{% endfor %}
