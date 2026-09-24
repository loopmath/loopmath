#!/usr/bin/env python3
"""Create reproducible broken HTML copies for visualizer probe audits."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = {
    "delete-nested": """
const target = DATA.nodes.find(node => node.spawn && Object.hasOwn(node.spawn, 'tool_use_id'));
if (!target) throw new Error('no spawn record for nested deletion');
delete target.spawn.tool_use_id;
""",
    "corrupt-nested": """
const target = DATA.nodes.find(node => node.launch && Object.hasOwn(node.launch, 'evidence'));
if (!target) throw new Error('no launch record for nested corruption');
target.launch.evidence = 'planted corrupted nested value';
""",
    "delete-force-link": """
const target = DATA.artifacts.find(artifact => artifact.c.length && artifact.t != null && artifact.w.length);
if (!target) throw new Error('no force artifact connection for deletion');
target.w.shift(); target.we.shift();
""",
    "delete-field": """
const target = DATA.nodes.reduce((best, node) => node.written.length + node.read.length > best.written.length + best.read.length ? node : best);
delete target.pt;
""",
    "delete-token": """
const target = DATA.nodes.reduce((best, node) => node.written.length + node.read.length > best.written.length + best.read.length ? node : best);
delete target.tok.cache_write_5m;
""",
    "browser-wreck": """
console.error('planted browser failure');
setTimeout(() => { throw new Error('planted uncaught failure'); }, 0);
const image = document.createElement('img'); image.src = 'https://example.com/planted.png'; document.body.appendChild(image);
for (const key of ['swim', 'cost']) {
  const box = document.querySelector('[data-view="' + key + '"] [data-view-artifacts]');
  box.checked = true; box.dispatchEvent(new Event('change', { bubbles: true }));
}
document.querySelector('[data-view="force"] [data-node]').dispatchEvent(new MouseEvent('click', { bubbles: true }));
document.querySelectorAll('[data-art]').forEach(el => el.remove());
document.querySelectorAll('.hollow').forEach(el => el.classList.remove('hollow'));
document.querySelectorAll('[data-view="cost"] text').forEach(el => { el.textContent = el.textContent.replace('(untimed)', '(dirty)'); });
document.querySelector('#table th').textContent = 'wrong column';
const views = [...document.querySelectorAll('details[data-view]')];
views[0].open = false; views[0].parentElement.appendChild(views[1]);
views[0].parentElement.appendChild(document.querySelector('[data-attempt-table]'));
for (const summary of document.querySelectorAll('details[data-view] summary')) summary.addEventListener('click', event => event.preventDefault());
document.querySelector('#table').replaceWith(document.querySelector('#table').cloneNode(true));
document.querySelector('#filters').replaceWith(document.querySelector('#filters').cloneNode(true));
for (const controls of [...document.querySelectorAll('[data-controls]')]) controls.replaceWith(controls.cloneNode(true));
for (const host of [...document.querySelectorAll('[data-graph]')]) host.replaceWith(host.cloneNode(true));
document.querySelector('#accounting').textContent = 'dirty accounting';
const card = document.querySelector('#card'); card.hidden = false; card.innerHTML = '<span data-art-link="0">dirty artifact link</span>';
""",
    "parity-wreck": """
const dirty = value => value.setAttribute('data-dirty', 'yes');
document.querySelector('[data-view="force"] [data-node]').dispatchEvent(new MouseEvent('click', { bubbles: true }));
for (const row of document.querySelectorAll('#table tbody tr')) row.dataset.i += '-dirty';
for (const cell of document.querySelectorAll('#table th,#table td')) cell.textContent += '!';
document.querySelector('#table').replaceWith(document.querySelector('#table').cloneNode(true));
document.querySelector('#filters').replaceWith(document.querySelector('#filters').cloneNode(true));
for (const controls of [...document.querySelectorAll('[data-controls]')]) controls.replaceWith(controls.cloneNode(true));
const views = [...document.querySelectorAll('details[data-view]')];
views[0].open = false;
views[0].parentElement.appendChild(views[1]);
views[0].parentElement.appendChild(document.querySelector('[data-attempt-table]'));
for (const view of views) {
  const host = view.querySelector('[data-graph]');
  host.querySelector('svg').setAttribute('width', '1');
  host.querySelectorAll('[data-node],[data-edge],[data-art],.alink,text,rect,line,path,circle').forEach(dirty);
  const ns = 'ht' + 'tp:' + '/' + '/www.w3.org/2000/svg';
  const extraArtifact = document.createElementNS(ns, 'circle');
  extraArtifact.setAttribute('data-art', '-99'); host.querySelector('svg').appendChild(extraArtifact);
  const extraLink = document.createElementNS(ns, 'path');
  extraLink.setAttribute('class', 'alink'); host.querySelector('svg').appendChild(extraLink);
  for (const selector of ['.e-spawn', '.e-launch', '.e-artifact.t-verified', '.e-artifact.t-heuristic', '.node', '.node.hollow,.seg.unavailable']) {
    const el = host.querySelector(selector); if (el) el.style.stroke = 'rgb(255, 0, 255)';
  }
  host.replaceWith(host.cloneNode(true));
}
for (const link of document.querySelectorAll('[data-view="force"] .alink')) link.removeAttribute('data-edge-list');
const forceControl = document.querySelector('[data-view="force"] [data-controls]');
forceControl.insertAdjacentHTML('beforeend', '<input data-view-artifacts>');
document.querySelector('#accounting').textContent = 'dirty accounting';
const card = document.querySelector('#card');
card.hidden = false;
card.innerHTML = '<span data-art-link="0">dirty artifact link</span>';
card.addEventListener('click', event => {
  if (!event.target.closest('[data-art-link]')) return;
  const artifact = DATA.artifacts[Number(event.target.closest('[data-art-link]').dataset.artLink)];
  artifact.hint = artifact.hint ? null : 'dirty hint';
  const term = card.querySelector('dt');
  if (term) { term.nextElementSibling?.remove(); term.remove(); }
  card.insertAdjacentHTML('beforeend', '<dt>hint</dt><dd>planted hint</dd>');
});
""",
}


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: graph_html_dirty_fixture.py SOURCE DESTINATION MODE", file=sys.stderr)
        return 2
    source, destination, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    page = source.read_text(encoding="utf-8")
    if mode == "static-wreck":
        page = page.replace("label: 'attempt'", "label: 'wrong column'", 1)
        page = page.replace(
            '<details class="viewsec" data-view="swim" open>',
            '<details class="viewsec" data-view="wrong">',
            1,
        )
        page = page.replace(
            "</body>",
            '<script src="https://example.com/x.js"></script><script>fetch("https://example.com/y")</script></body>',
            1,
        )
        page += "x" * max(0, 520_000 - len(page.encode("utf-8")))
    elif mode == "prototype-full-links":
        old = "const artList = (ids, limit = 8)"
        if old not in page:
            print("source does not contain the prototype card limit", file=sys.stderr)
            return 2
        page = page.replace(old, "const artList = (ids, limit = 1000)", 1)
    elif mode in SCRIPTS:
        page = page.replace("</body>", f"<script>{SCRIPTS[mode]}</script></body>", 1)
    else:
        print(f"unknown mode: {mode}", file=sys.stderr)
        return 2
    destination.write_text(page, encoding="utf-8")
    print(f"wrote {destination} with {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
