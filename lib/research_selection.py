"""Research-node selection (placeholder: uniform random).

Picks the next active G node to work on in the research / refinement loop. The
orchestrator calls this once per work unit and skips ids already chosen in the
current batch, so returning a single id (or None when nothing is active) is all
that is required.

Swappable: replace `make_research_select` with a smarter policy later (e.g.
favoring G nodes with resumable incomplete research, fewer attempts, or higher
expected value). The interface — `select(tree) -> g_node_id | None` — is stable.
"""

from __future__ import annotations

import random

from tree import Tree


def make_research_select(seed: int | None = None):
    """Return a `select(tree) -> g_node_id | None` closure (seeded for repro)."""
    rng = random.Random(seed)

    def research_select(tree: Tree) -> str | None:
        actives = tree.g_nodes(active_only=True)
        if not actives:
            return None
        return rng.choice(actives).node_id

    return research_select
