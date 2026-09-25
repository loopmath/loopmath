"""The shape builder: front (none, plan) + middle (implement, best_of_n, team) + back (none, review).

The six catalog shapes are named combinations; every other combination gets a
composed id (`team`, `plan_team`, `best_of_n_review`, ...). One-step edits and
`infer` both go through this builder, so an edit that removes the reviewer from
`implement_review` lands exactly on `solo`, with the same configuration id.

Every shape reads the inputs `issue` and `repo` and writes the output `diff`.
Workflows are acyclic: a repair loop is `control`, never an edge.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..types import Control, Gate, Piece, Workflow

MIDDLES = ("implement", "best_of_n", "team")
MIN_WIDTH, MAX_WIDTH = 2, 6
MIN_ROUNDS, MAX_ROUNDS = 1, 6
DEFAULT_WIDTH = 3
DEFAULT_REVIEW_ROUNDS = 3

# (plan, middle, review) -> catalog id
CATALOG_NAMES: dict[tuple[bool, str, bool], str] = {
    (False, "implement", False): "solo",
    (False, "best_of_n", False): "best_of_n",
    (True, "implement", False): "plan_implement",
    (False, "implement", True): "implement_review",
    (True, "implement", True): "plan_implement_review",
    (True, "team", True): "swarm",
}
CATALOG_IDS: tuple[str, ...] = ("solo", "best_of_n", "plan_implement", "implement_review",
                                "plan_implement_review", "swarm")

_TITLES = {
    "solo": "One agent implements",
    "best_of_n": "Best of n: parallel implementers, a referee picks one",
    "plan_implement": "Plan, then implement",
    "implement_review": "Implement, then review",
    "plan_implement_review": "Plan, implement, review",
    "swarm": "Swarm: a planner splits the work, parallel workers, a reviewer",
}
_PART_TITLES = {"plan": "plan", "implement": "implement", "best_of_n": "best of n",
                "team": "team of parallel workers", "review": "review"}

GATE_RULE_DEFAULTS = {"reviewer": "review_approve", "referee": "referee_pick", "tester": "tests_pass"}


@dataclass(frozen=True)
class ShapeParams:
    plan: bool = False
    middle: str = "implement"  # implement | best_of_n | team
    width: int = 1  # 1 for implement; 2 to 6 for best_of_n and team
    review: bool = False
    budget_rounds: int = 1  # K_max counting the first round; 1 without a review gate
    rescue: str = "redo_usual"

    def replace(self, **changes) -> "ShapeParams":
        return normalize(dataclasses.replace(self, **changes))


def normalize(p: ShapeParams) -> ShapeParams:
    """Fill the implied values: width 1 for `implement`, default width for the others, 1 round without review."""
    width = 1 if p.middle == "implement" else (p.width if p.width >= MIN_WIDTH else DEFAULT_WIDTH)
    rounds = p.budget_rounds if p.review else 1
    return dataclasses.replace(p, width=width, budget_rounds=max(1, rounds))


def shape_name(p: ShapeParams) -> str:
    key = (p.plan, p.middle, p.review)
    if key in CATALOG_NAMES:
        return CATALOG_NAMES[key]
    parts = (["plan"] if p.plan else []) + [p.middle] + (["review"] if p.review else [])
    return "_".join(parts)


def shape_title(p: ShapeParams) -> str:
    name = shape_name(p)
    if name in _TITLES:
        return _TITLES[name]
    parts = (["plan"] if p.plan else []) + [p.middle] + (["review"] if p.review else [])
    text = ", ".join(_PART_TITLES[x] for x in parts)
    return text[:1].upper() + text[1:]


def work_piece(p: ShapeParams) -> str:
    """The piece that produces the change: `work` for a team, else `implement`."""
    return "work" if p.middle == "team" else "implement"


def build_shape(params: ShapeParams) -> Workflow:
    p = normalize(params)
    if p.middle not in MIDDLES:
        raise ValueError(f"unknown middle {p.middle!r}; expected one of {', '.join(MIDDLES)}")
    pieces: list[Piece] = []
    kinds: dict[str, str] = {"issue": "issue", "repo": "repo"}
    edges: list[tuple[str, str]] = []
    gates: list[Gate] = []

    if p.plan:
        pieces.append(Piece("plan", "planner"))
        kinds["plan_doc"] = "plan"
        edges += [("issue", "plan"), ("repo", "plan"), ("plan", "plan_doc")]

    worker = work_piece(p)
    role = "worker" if p.middle == "team" else "implementer"
    pieces.append(Piece(worker, role, width=p.width))
    edges += [("issue", worker), ("repo", worker)]
    if p.plan:
        edges.append(("plan_doc", worker))
    if p.middle == "best_of_n":
        kinds["candidates"] = "diff"
        edges.append((worker, "candidates"))
        pieces.append(Piece("select", "referee"))
        edges += [("candidates", "select"), ("issue", "select"), ("select", "diff")]
        gates.append(Gate("g_select", "select", GATE_RULE_DEFAULTS["referee"]))
    else:
        edges.append((worker, "diff"))
    kinds["diff"] = "diff"

    if p.review:
        pieces.append(Piece("review", "reviewer"))
        kinds["verdict"] = "verdict"
        edges += [("diff", "review"), ("issue", "review")]
        if p.plan:
            edges.append(("plan_doc", "review"))
        edges.append(("review", "verdict"))
        gates.append(Gate("g_review", "review", GATE_RULE_DEFAULTS["reviewer"], on_fail=worker))

    order = ["issue", "repo", "plan_doc", "candidates", "diff", "verdict"]
    artifacts = tuple(a for a in order if a in kinds)
    return Workflow(
        id=shape_name(p), version=1, title=shape_title(p), pieces=tuple(pieces), artifacts=artifacts,
        edges=tuple(edges), control=Control(gates=tuple(gates), budget_rounds=p.budget_rounds, rescue=p.rescue),
        extra={"artifact_kinds": {a: kinds[a] for a in artifacts}},
    )


def guess_params(workflow: Workflow) -> ShapeParams | None:
    """Read the parameters a workflow would have if the builder made it (not yet checked)."""
    ids = {pc.id: pc for pc in workflow.pieces}
    if "work" in ids and "implement" in ids:
        return None
    if "select" in ids:
        middle = "best_of_n"
    elif "work" in ids:
        middle = "team"
    else:
        middle = "implement"
    main = ids.get("work") or ids.get("implement")
    if main is None:
        return None
    return normalize(ShapeParams(plan="plan" in ids, middle=middle, width=main.width, review="review" in ids,
                                 budget_rounds=workflow.control.budget_rounds, rescue=workflow.control.rescue))


def shape_params(workflow: Workflow) -> ShapeParams | None:
    """The builder parameters of `workflow`, or None when the builder cannot make it exactly."""
    from .ids import structure_key

    p = guess_params(workflow)
    if p is None:
        return None
    try:
        built = build_shape(p)
    except ValueError:
        return None
    return p if structure_key(built) == structure_key(workflow) else None


def catalog_shapes() -> dict[str, Workflow]:
    """The six catalog shapes straight from the builder (the TOML files must equal these)."""
    out: dict[str, Workflow] = {}
    for (plan, middle, review), name in CATALOG_NAMES.items():
        width = DEFAULT_WIDTH if middle != "implement" else 1
        rounds = DEFAULT_REVIEW_ROUNDS if review else 1
        out[name] = build_shape(ShapeParams(plan=plan, middle=middle, width=width, review=review, budget_rounds=rounds))
    return {k: out[k] for k in CATALOG_IDS}


def nearest_catalog(p: ShapeParams) -> str:
    """The catalog shape closest to a composed one.

    Distance: 1 per planner or reviewer difference, 2 between the two parallel
    middles, 3 between `implement` and a parallel middle. Ties go to the shape
    with the same middle, then the smaller shape.
    """
    best = None
    for (plan, middle, review), name in CATALOG_NAMES.items():
        if middle == p.middle:
            mid = 0
        elif "implement" in (middle, p.middle):
            mid = 3
        else:
            mid = 2
        dist = (plan != p.plan) + mid + (review != p.review)
        size = plan + review + (middle != "implement")
        key = (dist, middle != p.middle, size, CATALOG_IDS.index(name))
        if best is None or key < best[0]:
            best = (key, name)
    return best[1]
