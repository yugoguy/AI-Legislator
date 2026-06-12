"""Parliament portfolio selection: rank active 議案 by Evaluator quality.

Returns the active G nodes ordered best-first by their latest Evaluator score Q
(stats["Q"], written by the research-selection policy). The orchestrator keeps
the first `parliament_max` and closes the rest, so this ordering decides which
proposals are scrutinized in parliament.

Unlike research selection there is no exploration term here — parliament is a
one-shot commitment of effort to the most promising proposals, so pure Q ranking
is appropriate. Ties (e.g. Gs never scored, Q=0) fall back to a seeded shuffle so
the order is deterministic but not biased by node id.

Swappable: the interface `select(tree) -> list[g_node_id]` is stable.
"""

from __future__ import annotations

import random

from tree import Tree


def make_parliament_select(seed: int | None = None):
    """Return a `select(tree) -> list[g_node_id]` closure (seeded for ties)."""
    rng = random.Random(seed)

    def parliament_select(tree: Tree) -> list[str]:
        actives = tree.g_nodes(active_only=True)
        # Seeded jitter breaks ties without depending on node id ordering.
        decorated = [(g.q_score, rng.random(), g.node_id) for g in actives]
        decorated.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [node_id for _, _, node_id in decorated]

    return parliament_select
