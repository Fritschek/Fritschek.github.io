---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

An up-to-date list of my articles can be found on <u><a href="https://scholar.google.com/citations?user=EfwPnJQAAAAJ&hl=en">my Google Scholar profile</a></u> and my <u><a href="https://tu-dresden.de/ing/elektrotechnik/ifn/itml/die-professur/team/fritschek?set_language=en">TU Dresden FIS profile</a></u>.

{% include base_path %}

<style>
.publication-list {
  counter-reset: publication;
  list-style: none;
  margin-left: 0;
  padding-left: 0;
}

.publication-list > li {
  counter-increment: publication;
  margin-bottom: 0.75em;
  padding-left: 2.35em;
  position: relative;
}

.publication-list > li::before {
  content: counter(publication) ".";
  left: 0;
  position: absolute;
  text-align: right;
  width: 1.7em;
}
</style>

## Journal Articles

<ol class="publication-list">
{% for pub in site.data.publications.journal %}
  {% assign authors = pub.authors | markdownify | remove: '<p>' | remove: '</p>' | strip %}
  <li>{% if pub.id %}<span id="{{ pub.id }}"></span>{% endif %}{{ authors }}, "{{ pub.title }}"{% if pub.venue %}, in <em>{{ pub.venue }}</em>{% endif %}{% if pub.details %}, {{ pub.details }}{% endif %}.{% if pub.links %}{% for link in pub.links %} [<a href="{{ link.url }}">{{ link.label }}</a>]{% endfor %}{% endif %}</li>
{% endfor %}
</ol>

## Conference Articles

<ol class="publication-list">
{% for pub in site.data.publications.conference %}
  {% assign authors = pub.authors | markdownify | remove: '<p>' | remove: '</p>' | strip %}
  <li>{% if pub.id %}<span id="{{ pub.id }}"></span>{% endif %}{{ authors }}, "{{ pub.title }}"{% if pub.venue %}, in <em>{{ pub.venue }}</em>{% endif %}{% if pub.details %}, {{ pub.details }}{% endif %}.{% if pub.links %}{% for link in pub.links %} [<a href="{{ link.url }}">{{ link.label }}</a>]{% endfor %}{% endif %}</li>
{% endfor %}
</ol>

## Book Chapters

<ol class="publication-list">
{% for pub in site.data.publications.book_chapters %}
  {% assign authors = pub.authors | markdownify | remove: '<p>' | remove: '</p>' | strip %}
  <li>{% if pub.id %}<span id="{{ pub.id }}"></span>{% endif %}{{ authors }}, "{{ pub.title }}"{% if pub.venue %}, in <em>{{ pub.venue }}</em>{% endif %}{% if pub.details %}, {{ pub.details }}{% endif %}.{% if pub.links %}{% for link in pub.links %} [<a href="{{ link.url }}">{{ link.label }}</a>]{% endfor %}{% endif %}</li>
{% endfor %}
</ol>

## Preprints

<ol class="publication-list">
{% for pub in site.data.publications.preprints %}
  {% assign authors = pub.authors | markdownify | remove: '<p>' | remove: '</p>' | strip %}
  <li>{% if pub.id %}<span id="{{ pub.id }}"></span>{% endif %}{{ authors }}, "{{ pub.title }}"{% if pub.venue %}, in <em>{{ pub.venue }}</em>{% endif %}{% if pub.details %}, {{ pub.details }}{% endif %}.{% if pub.links %}{% for link in pub.links %} [<a href="{{ link.url }}">{{ link.label }}</a>]{% endfor %}{% endif %}</li>
{% endfor %}
</ol>
