"""Forest nodes and parents: task chain, provider/family/version, effort, role, topology, features, source.

Spec 04 section 1. Every row of data carries a sparse set of nodes. A node's effect is
Gaussian around its parent's effect with the scale `phi` of its level (the tree law).
The engine uses the equivalent additive form: a row carries every node on each of its
chains, and every node is an independent deviation `N(0, phi_level^2)`. A node without
data then predicts its parent's belief widened by `phi`, as the tree law says.

Node ids are `level:key`. Keys nest along a chain so that each node has one parent:
`type:feature`, `repo:feature/owner/name`, `subtype:feature/owner/name/api`, `task:<id>`;
`provider:anthropic`, `family:opus`, `model:claude-opus-5-5`, `fsrc:opus|sweep`; `effort:high`,
`family_effort:opus|high`; `role:reviewer`, `role_family:reviewer|opus`;
`topology:implement_review`, `position:implement_review#1`, `psrc:implement_review#1|sweep`;
`feature:size=s`;
`source:sweep`; `org:<org>` (crossed, since org is optional); `gate:<rule>`.
Fixed effects (wide priors, no scale fit): `fixed:intercept`, `round:2`, `round:3+`,
`control:budget`, `control:width`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

# Default level scales, spec 04 section 4: (cost head on log usd, logistic heads, score heads).
# Score scales are in units of the score's standard deviation (score responses are standardized).
DEFAULT_SCALES: dict[str, tuple[float, float, float]] = {
    "org": (0.5, 1.0, 1.0),
    "type": (0.5, 1.0, 1.0),
    "repo": (0.4, 0.7, 0.7),
    "subtype": (0.3, 0.5, 0.5),
    "task": (0.5, 1.0, 1.0),
    "provider": (0.7, 1.0, 1.0),
    "family": (0.7, 1.0, 1.0),
    "model": (0.3, 0.5, 0.5),
    "effort": (0.4, 0.5, 0.5),
    "family_effort": (0.2, 0.3, 0.3),
    "harness": (0.3, 0.5, 0.5),
    "role": (0.5, 0.5, 0.5),
    "role_family": (0.3, 0.5, 0.5),
    "topology": (0.5, 0.7, 0.7),
    "position": (0.2, 0.3, 0.3),
    "psrc": (0.7, 0.7, 0.7),  # position x source, on the cost and tokens heads only
    "fsrc": (0.5, 0.5, 0.5),  # family x source, on the cost and tokens heads only
    "feature": (0.3, 0.4, 0.4),
    "source": (0.3, 0.5, 0.5),
    "gate": (0.5, 0.7, 0.7),
}
HEAD_KIND_COLUMN = {"cost": 0, "logit": 1, "score": 2}

# Fixed effects: prior standard deviation per head kind (wide on purpose).
FIXED_SD: dict[str, tuple[float, float, float]] = {
    "fixed": (10.0, 5.0, 5.0),
    "round": (2.0, 2.0, 2.0),
    "control": (1.0, 1.0, 1.0),
    "horizon": (10.0, 10.0, 10.0),  # wide: the informative prior is a FactorSpec in fit() (spec 04 section 1)
}

HYPER_SD = 0.7  # sd of the log-normal hyperprior on each phi, around the defaults above

# Parent level of each chained level. Levels absent here have no parent.
PARENT_LEVEL = {
    "repo": "type",
    "subtype": "repo",
    "family": "provider",
    "model": "family",
    "family_effort": "effort",
    "role_family": "role",
    "position": "topology",
    "psrc": "position",
    "fsrc": "family",
}

# Level names the posterior command accepts, and the node levels each one shows.
LEVEL_ALIASES = {
    "model": ("provider", "family", "model", "fsrc"),
    "effort": ("effort", "family_effort"),
    "role": ("role", "role_family"),
    "topology": ("topology", "position", "psrc"),
}

ROLE_WORDS = ("planner", "implementer", "reviewer", "tester", "referee", "worker")


def canonical_role(role: str | None) -> str:
    """Fold a role word into the types.py vocabulary through `workflows.normalize_role()` (lane 4);
    no role is `worker`."""
    return _canonical_role(str(role or "").strip().lower())


@lru_cache(maxsize=4096)
def _canonical_role(raw: str) -> str:
    from ..workflows import normalize_role

    return str(normalize_role(raw) or "worker")


_ANTHROPIC_FAMILIES = ("opus", "sonnet", "fable", "haiku", "mythos")
# known families with or without the `claude-` prefix; any other `claude-<family>-<n>` is a new family
_CLAUDE_RE = re.compile(r"^(?:claude-(?=[a-z]+-\d)([a-z]+)|(" + "|".join(_ANTHROPIC_FAMILIES) + r"))"
                        r"-(\d+)(?:[.-](\d+))?(?:-\d{8})?$")
_GPT_RE = re.compile(r"^gpt-(\d+(?:\.\d+)?)(?:-(.+))?$")


def canonical_model_id(model: str | None) -> str:
    """One spelling per model: `claude-opus-5-5`, `claude-haiku-4-5`, `gpt-6-astra`.

    Passes through `ingest.base.canonical_model` first (lane 2 owns its aliases), then
    rewrites Claude ids to the `claude-<family>-<major>[-<minor>]` form of spec 04.
    """
    from ..ingest.base import canonical_model

    raw = str(model or "").strip().lower()
    raw = canonical_model(raw) or raw
    m = _CLAUDE_RE.match(raw)
    if m:
        fam, major, minor = m.group(1) or m.group(2), m.group(3), m.group(4)
        return f"claude-{fam}-{major}" + (f"-{minor}" if minor else "")
    return raw or "unknown"


def model_path(model: str) -> tuple[str, str, str]:
    """(provider, family, version) for a model id; version is the canonical id.

    `claude-opus-5` and `claude-opus-5-5` are (anthropic, opus); `gpt-5.6-sol` and
    `gpt-6-sol` are (openai, sol); `gpt-6-astra` is (openai, astra). A new version starts
    from its family, and a new family from its provider.
    """
    mid = canonical_model_id(model)
    m = _CLAUDE_RE.match(mid)
    if m:
        return ("anthropic", m.group(1) or m.group(2), mid)
    m = _GPT_RE.match(mid)
    if m:
        return ("openai", m.group(2) or "gpt", mid)
    if re.match(r"^o\d", mid) or mid.startswith("codex"):
        return ("openai", mid.split("-")[0], mid)
    for prefix, provider in (("gemini", "google"), ("deepseek", "deepseek"), ("grok", "xai"),
                             ("qwen", "alibaba"), ("kimi", "moonshot"), ("glm", "zhipu"),
                             ("mistral", "mistral"), ("llama", "meta")):
        if mid.startswith(prefix):
            return (provider, prefix, mid)
    head = mid.split("-")[0] if mid else "unknown"
    return ("other", head or "unknown", mid)


def level_of(node_id: str) -> str:
    return node_id.split(":", 1)[0]


def scale_group(node_id: str) -> str | None:
    """The phi group of a node: its level, or `feature:<key>` for features; None for fixed effects."""
    level = level_of(node_id)
    if level in FIXED_SD:
        return None
    if level == "feature":
        key = node_id.split(":", 1)[1].split("=", 1)[0]
        return f"feature:{key}"
    return level


def default_scale(group: str, kind: str) -> float:
    base = group.split(":", 1)[0]
    return DEFAULT_SCALES.get(base, DEFAULT_SCALES["feature"])[HEAD_KIND_COLUMN[kind]]


def fixed_sd(node_id: str, kind: str) -> float:
    return FIXED_SD[level_of(node_id)][HEAD_KIND_COLUMN[kind]]


def display_level(node_id: str) -> str:
    """The level shown in NodeSummary: `feature:<key>` for features, else the node's level."""
    group = scale_group(node_id)
    return group if group else level_of(node_id)


def node_key(node_id: str) -> str:
    """The human key of a node: its key without the nesting prefix."""
    level, _, key = node_id.partition(":")
    if level in ("repo", "subtype"):
        return key.split("/", 1)[1] if "/" in key else key
    return key


@dataclass
class Forest:
    """Node registry shared by every head: id and parent per node."""

    ids: list[str] = field(default_factory=list)
    parents: dict[str, str | None] = field(default_factory=dict)
    index: dict[str, int] = field(default_factory=dict)

    def add(self, node_id: str, parent: str | None = None) -> int:
        idx = self.index.get(node_id)
        if idx is None:
            idx = len(self.ids)
            self.ids.append(node_id)
            self.index[node_id] = idx
            self.parents[node_id] = parent
        elif parent and not self.parents.get(node_id):
            self.parents[node_id] = parent
        return idx

    def to_json(self) -> list[dict]:
        return [{"id": n, "level": display_level(n), "key": node_key(n), "parent": self.parents.get(n),
                 "group": scale_group(n)} for n in self.ids]

    @classmethod
    def from_json(cls, nodes: list[dict]) -> "Forest":
        f = cls()
        for n in nodes:
            f.add(n["id"], n.get("parent"))
        return f

    def ancestors(self, node_id: str) -> list[str]:
        """Parent chain of a node, nearest first."""
        out = []
        cur = self.parents.get(node_id)
        while cur:
            out.append(cur)
            cur = self.parents.get(cur)
        return out
