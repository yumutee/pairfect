"""Exhaustive, exact search over pairs of Wordle opening words.

A pair of openers is played blind: both words are guessed before looking at
either feedback. Together they split the candidate answers into cells that
share the same pair of feedbacks, and every metric below scores that split.

The search visits every unordered pair of openers. For the metrics that can
be bounded by single-word entropy it prunes with, so a pair is skipped only
when it provably cannot reach the current top k.

The results are exact either way.
"""

import math
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from pairfect import kernels
from pairfect.words import encode


@dataclass(frozen=True)
class Metric:
    name: str
    title: str
    description: str
    prunable: bool


METRICS = {
    m.name: m
    for m in [
        Metric(
            "entropy",
            "most information",
            "expected information from the two feedbacks, in bits (higher is better)",
            prunable=True,
        ),
        Metric(
            "outcomes",
            "most distinct outcomes",
            "number of distinct feedback combinations; divided by the number of"
            " answers, it is the chance of winning by guess 3 (higher is better)",
            prunable=False,
        ),
        Metric(
            "expected",
            "fewest answers left on average",
            "expected number of answers still possible after both guesses"
            " (lower is better)",
            prunable=True,
        ),
        Metric(
            "worst",
            "smallest worst case",
            "largest number of answers that can still be possible after both"
            " guesses (lower is better)",
            prunable=False,
        ),
    ]
}

BLOCK = 128  # first words scored per kernel call
SEED = 256  # top single words searched first to get a pruning threshold

_ROW = np.dtype(
    [
        ("key", "f8"),
        ("a", "i8"),
        ("b", "i8"),
        ("xsum", "f8"),
        ("outcomes", "i8"),
        ("sumsq", "i8"),
        ("worst", "i8"),
        ("singles", "i8"),
    ]
)


@dataclass(frozen=True)
class PairStats:
    words: tuple[str, str]
    entropy: float  # bits
    outcomes: int  # distinct feedback combinations
    expected: float  # expected number of answers left
    worst: int  # most answers that can be left
    singles: int  # answers identified exactly by the two feedbacks
    n_answers: int

    @property
    def win_by_3(self) -> float:
        """Chance of winning by guess 3 if guess 3 is any remaining candidate."""
        return self.outcomes / self.n_answers


@dataclass(frozen=True)
class Evaluation:
    """Guesses needed per answer: the pair, then greedy follow-up guesses."""

    words: tuple[str, str]
    histogram: dict[int, int]  # guesses needed -> number of answers

    @property
    def mean(self) -> float:
        total = sum(self.histogram.values())
        return sum(k * v for k, v in self.histogram.items()) / total

    @property
    def failures(self) -> int:
        """Answers that would take more than Wordle's six guesses."""
        return sum(v for k, v in self.histogram.items() if k > 6)


@dataclass(frozen=True)
class SearchResult:
    boards: dict[str, list[PairStats]]
    openers: int
    pairs_total: int
    pairs_scored: int
    seconds: float


def _key(metric: str, rows: dict[str, np.ndarray], n: int) -> np.ndarray:
    """Ranking key for `metric`: larger is better.

    Ties on the primary measure are broken by entropy, which is scaled into
    [0, 1) so it never outweighs a whole unit of an integer measure.
    """
    log2n = math.log2(n)
    entropy = log2n - rows["xsum"] / n
    if metric == "entropy":
        return entropy
    tie = entropy / (log2n + 1)
    if metric == "outcomes":
        return rows["outcomes"] + tie
    if metric == "expected":
        return tie - rows["sumsq"]
    if metric == "worst":
        return tie - rows["sumsq"] - rows["worst"] * (float(n) * n + 1)
    raise ValueError(f"unknown metric {metric!r}")


class _Board:
    """The best k pairs seen so far for one metric."""

    def __init__(self, metric: str, k: int):
        self.metric = metric
        self.k = k
        self.rows = np.empty(0, _ROW)

    def full(self) -> bool:
        return len(self.rows) >= self.k

    def threshold(self) -> float:
        return self.rows["key"][-1] if self.full() else -np.inf

    def entropy_floor(self, n: int) -> float:
        """Joint entropy a pair must have to possibly make this board."""
        if not self.full() or not METRICS[self.metric].prunable:
            return -np.inf
        if self.metric == "entropy":
            return self.rows["key"][-1]
        # trivial: sumsq <= S_k needs N**2 * 2**-H <= S_k
        return 2 * math.log2(n) - math.log2(self.rows["sumsq"][-1])

    def offer(self, rows: np.ndarray) -> None:
        if len(rows) > 4 * self.k:
            # Keep everything tied with the k-th best so tie-breaks stay exact.
            cut = np.partition(rows["key"], len(rows) - self.k)[len(rows) - self.k]
            rows = rows[rows["key"] >= cut]
        merged = np.concatenate([self.rows, rows])
        order = np.lexsort((merged["b"], merged["a"], -merged["key"]))
        self.rows = merged[order[: self.k]]


class Engine:
    def __init__(self, answers: Iterable[str], guesses: Iterable[str]):
        self.answers = sorted(set(answers))
        # Every answer is always an acceptable guess.
        self.guesses = sorted(set(guesses) | set(self.answers))
        self._index = {w: i for i, w in enumerate(self.guesses)}
        self.n = len(self.answers)
        self.P = kernels.pattern_matrix(encode(self.guesses), encode(self.answers))
        counts = np.arange(self.n + 1, dtype=np.float64)
        self.xlog2x = counts * np.log2(np.maximum(counts, 1))
        self.H = kernels.single_entropy(self.P, self.xlog2x)
        self.answer_guess = np.array([self._index[w] for w in self.answers])

    def index(self, word: str) -> int:
        try:
            return self._index[word.lower()]
        except KeyError:
            raise KeyError(f"{word!r} is not in the guess list") from None

    def pool(self, name: str) -> np.ndarray:
        """Guess indices allowed as openers: 'all' guesses or only 'answers'."""
        if name == "all":
            return np.arange(len(self.guesses))
        if name == "answers":
            return self.answer_guess.copy()
        raise ValueError(f"unknown pool {name!r}")

    def _stats(self, row: np.void) -> PairStats:
        a, b = int(row["a"]), int(row["b"])
        if self.H[b] > self.H[a]:
            a, b = b, a
        return PairStats(
            words=(self.guesses[a], self.guesses[b]),
            entropy=math.log2(self.n) - row["xsum"] / self.n,
            outcomes=int(row["outcomes"]),
            expected=row["sumsq"] / self.n,
            worst=int(row["worst"]),
            singles=int(row["singles"]),
            n_answers=self.n,
        )

    def score(self, first: str, second: str) -> PairStats:
        a, b = self.index(first), self.index(second)
        firsts = np.array([a])
        stats = kernels.pair_stats(self.P, firsts, np.array([b]), np.array([b + 1]), self.xlog2x)
        rows = self._records(stats, firsts, np.arange(len(self.guesses)), stats[1] > 0)
        return self._stats(rows[0])

    def _records(self, stats, firsts, ids, keep) -> np.ndarray:
        """One _ROW record (key unset) per pair where `keep` is true.

        `stats` comes from kernels.pair_stats; `firsts` are its row indices into
        P and `ids` maps rows of P to guess indices.
        """
        xsum, outcomes, sumsq, worst, singles = stats
        rr, cc = np.nonzero(keep)
        a, b = ids[firsts[rr]], ids[cc]
        out = np.empty(len(rr), _ROW)
        out["a"], out["b"] = np.minimum(a, b), np.maximum(a, b)
        out["xsum"] = xsum[rr, cc]
        out["outcomes"] = outcomes[rr, cc]
        out["sumsq"] = sumsq[rr, cc]
        out["worst"] = worst[rr, cc]
        out["singles"] = singles[rr, cc]
        return out

    def search(
        self,
        metrics: Iterable[str] = ("entropy",),
        top: int = 10,
        pool: str = "all",
        partner: str | None = None,
        prune: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> SearchResult:
        """Find the best `top` pairs of openers for each metric.

        With `partner`, only pairs containing that word are considered.
        """
        if top < 1:
            raise ValueError("top must be at least 1")
        start = time.perf_counter()
        boards = {m: _Board(m, top) for m in metrics}
        for m in boards:
            if m not in METRICS:
                raise ValueError(f"unknown metric {m!r}")
        ids = self.pool(pool)

        if partner is not None:
            a = self.index(partner)
            ids = ids[ids != a]
            P = np.concatenate([self.P[[a]], self.P[ids]])
            firsts = np.array([0])
            stats = kernels.pair_stats(P, firsts, np.array([1]), np.array([len(P)]), self.xlog2x)
            rows = self._records(stats, firsts, np.concatenate([[a], ids]), stats[1] > 0)
            for m, board in boards.items():
                rows["key"] = _key(m, rows, self.n)
                board.offer(rows.copy())
            return self._result(boards, len(ids), len(ids), len(ids), start)

        # Visit first words in order of falling single-word entropy, so the
        # pairs that survive pruning are a prefix of each row.
        order = np.argsort(-self.H[ids], kind="stable")
        ids = ids[order]
        P = np.ascontiguousarray(self.P[ids])
        H = self.H[ids]
        G = len(ids)
        prunable = prune and all(METRICS[m].prunable for m in boards)

        floor = -np.inf
        if prunable and G > SEED:
            seed = {m: _Board(m, top) for m in boards}
            self._sweep(P[:SEED], H[:SEED], ids[:SEED], seed, -np.inf, False, None)
            floor = min(b.entropy_floor(self.n) for b in seed.values())
        scored = self._sweep(P, H, ids, boards, floor, prunable, progress)
        return self._result(boards, G, G * (G - 1) // 2, scored, start)

    def _sweep(self, P, H, ids, boards, floor, prunable, progress) -> int:
        G = len(ids)
        neg_h = -H
        scored = 0
        for i in range(0, G - 1, BLOCK):
            rows = np.arange(i, min(i + BLOCK, G - 1))
            lo = rows + 1
            hi = np.full(len(rows), G)
            if prunable:
                floor = max([floor] + [b.entropy_floor(self.n) for b in boards.values()])
                # Partners b need H[b] >= floor - H[a]; H is sorted descending.
                need = floor - H[rows] - 1e-9
                hi = np.maximum(np.searchsorted(neg_h, -need, side="right"), lo)
                if hi[0] == lo[0]:
                    break  # every later row is pruned too
            stats = kernels.pair_stats(P, rows, lo, hi, self.xlog2x)
            scored += int((hi - lo).sum())
            fields = dict(zip(("xsum", "outcomes", "sumsq", "worst"), stats))
            valid = stats[1] > 0
            for m, board in boards.items():
                key = _key(m, fields, self.n)
                keep = valid & (key >= board.threshold())
                recs = self._records(stats, rows, ids, keep)
                recs["key"] = key[keep]
                board.offer(recs)
            if progress:
                progress(rows[-1] + 1, G)
        return scored

    def _result(self, boards, openers, total, scored, start) -> SearchResult:
        return SearchResult(
            boards={m: [self._stats(r) for r in b.rows] for m, b in boards.items()},
            openers=openers,
            pairs_total=total,
            pairs_scored=scored,
            seconds=time.perf_counter() - start,
        )

    def evaluate(self, first: str, second: str) -> Evaluation:
        """Play `first` and `second`, then greedy follow-ups, against every answer."""
        a, b = self.index(first), self.index(second)
        hist: Counter[int] = Counter()
        joint = self.P[a].astype(np.int64) * kernels.N_CODES + self.P[b]
        for code in np.unique(joint):
            cell = np.flatnonzero(joint == code)
            if len(cell) == 1 and self.answer_guess[cell[0]] in (a, b):
                hist[1 if self.answer_guess[cell[0]] == a else 2] += 1
            else:
                self._solve(cell, 2, hist)
        return Evaluation((self.guesses[a], self.guesses[b]), dict(sorted(hist.items())))

    def _solve(self, cell: np.ndarray, used: int, hist: Counter) -> None:
        if len(cell) <= 2:
            # Guess a candidate: right now, or else the other one next.
            for k in range(len(cell)):
                hist[used + 1 + k] += 1
            return
        g = kernels.best_follow_up(self.P, cell, self.answer_guess[cell])
        codes = self.P[g, cell]
        for code in np.unique(codes):
            sub = cell[codes == code]
            if code == kernels.ALL_GREEN:
                hist[used + 1] += 1
            else:
                self._solve(sub, used + 1, hist)
