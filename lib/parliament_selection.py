"""Parliament portfolio selection (placeholder: random order).

Returns the active G nodes in random order as the parliament portfolio. The
orchestrator keeps the first `parliament_max` and closes every other active G,
so the order returned here is what decides the portfolio.

Swappable: replace `make_parliament_select` with a policy that ranks 議案 by
maturity, evidence strength, or diversity. The interface —
`select(tree) -> list[g_node_id]` — is stable.
"""

from __future__ import annotations

import random

from tree import Tree


def make_parliament_select(seed: int | None = None):
    """Return a `select(tree) -> list[g_node_id]` closure (seeded for repro)."""
    rng = random.Random(seed)

    def parliament_select(tree: Tree) -> list[str]:
        ids = [g.node_id for g in tree.g_nodes(active_only=True)]
        rng.shuffle(ids)
        return ids

    return parliament_select
