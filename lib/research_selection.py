"""Research-node selection: UCB over active 議案 (G nodes).

Replaces the previous uniform-random placeholder. The next active G to research
is chosen by a bandit-style score that balances proposal quality against how
little a G has been explored:

    score(g) = Q(g) + c * sqrt( ln N_i / n_i )

  - Q(g):  Evaluator quality in [0,1] (avg of five criteria; see evaluator.py).
           The exploitation term — promising proposals get more effort.
  - n_i:   research count of g (times it has been selected/researched). The
           uncertainty shrinks as a G accumulates research.
  - N_i:   selection rounds g has been PRESENT for = (current pick counter)
           - (g's birth step). Using a per-G "steps outstanding" rather than the
           global total means a newly spawned G is not penalized by history it
           never participated in.
  - c:     exploration weight (cfg.ucb_c; UCB1 standard is sqrt(2), default 1.0).

Scores are turned into a sampling distribution by plain normalization (NOT
softmax): P(g) = score(g) / sum_j score(g_j). With Q in [0,1] and c=1 the score
is always positive, so this is well-defined.

Modularity: ALL policy state and logic live here. The selector is a stateful
object exposing the stable `select(tree) -> g_id | None` interface the
orchestrator already expects. It owns the global pick counter and triggers the
Evaluator itself (injected), writing Q + history onto each G via the tree. Swap
this one file to change the selection algorithm; nothing else needs to know.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from tree import Tree
from node import GNode
from config import Config


class UCBResearchSelect:
    """Stateful UCB selector. Call instances like a function: select(tree)."""

    def __init__(self, cfg: Config, evaluator: Callable[..., tuple[float, dict]],
                 seed: int | None = None):
        self.cfg = cfg
        self.evaluator = evaluator          # evaluate(cfg, g, record=...) -> (Q, subs)
        self.rng = random.Random(seed)
        self.step = 0                       # global pick counter (advances per call)

    # -- bookkeeping kept inside the policy --------------------------------

    def _birth_step(self, g: GNode) -> int:
        """Step at which g first became eligible; stamped lazily on first sight."""
        bs = g.stats.get("birth_step")
        if bs is None:
            bs = self.step
            g.stats["birth_step"] = bs
        return int(bs)

    def _maybe_evaluate(self, tree: Tree, g: GNode) -> None:
        """Re-score g with the Evaluator every cfg.eval_every selections of it.

        Scores on first sight (no history yet) and whenever the research count has
        advanced by at least eval_every since the last scored count.
        """
        n = int(g.stats.get("research_count", 0))
        last = g.stats.get("q_eval_at_count")
        due = last is None or (n - int(last)) >= max(1, self.cfg.eval_every)
        if not due:
            return
        try:
            q, subs = self.evaluator(self.cfg, g, record=g.record_raw)
        except Exception:
            # Transient eval failure (API drop after retries): keep the prior Q
            # and let the next due round try again, rather than crashing selection.
            return
        g.record_score(self.step, subs, q)
        g.stats["q_eval_at_count"] = n
        tree.save(g)

    def _ucb(self, g: GNode) -> float:
        n_i = max(1, int(g.stats.get("research_count", 0)))
        N_i = max(2, self.step - self._birth_step(g) + 1)   # >=2 so ln N_i > 0
        explore = self.cfg.ucb_c * math.sqrt(math.log(N_i) / n_i)
        return g.q_score + explore

    # -- the stable interface ----------------------------------------------

    def __call__(self, tree: Tree) -> str | None:
        actives: list[GNode] = tree.g_nodes(active_only=True)
        if not actives:
            return None
        self.step += 1

        for g in actives:
            self._birth_step(g)
            self._maybe_evaluate(tree, g)

        scores = [max(self._ucb(g), 1e-9) for g in actives]
        total = sum(scores)
        r = self.rng.random() * total
        acc = 0.0
        for g, s in zip(actives, scores):
            acc += s
            if r <= acc:
                return g.node_id
        return actives[-1].node_id


def make_research_select(cfg: Config, evaluator, seed: int | None = None):
    """Return a stateful `select(tree) -> g_id | None` UCB selector."""
    return UCBResearchSelect(cfg, evaluator, seed=seed)
