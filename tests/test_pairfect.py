import json
import math
import random
from collections import Counter

import pytest

from pairfect import kernels, words
from pairfect.cli import main
from pairfect.engine import Engine


def reference_feedback(guess: str, answer: str) -> int:
    """Straightforward Wordle scoring, as a base-3 code (green 2, yellow 1)."""
    colors = [0] * 5
    unmatched = Counter(a for g, a in zip(guess, answer) if g != a)
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            colors[i] = 2
        elif unmatched[g]:
            colors[i] = 1
            unmatched[g] -= 1
    return sum(c * 3 ** i for i, c in enumerate(colors))


def reference_stats(first: str, second: str, answers: list[str], fb=reference_feedback) -> dict:
    cells = Counter((fb(first, x), fb(second, x)) for x in answers)
    n = len(answers)
    sizes = list(cells.values())
    return {
        "entropy": math.log2(n) - sum(s * math.log2(s) for s in sizes) / n,
        "outcomes": len(sizes),
        "expected": sum(s * s for s in sizes) / n,
        "worst": max(sizes),
        "singles": sizes.count(1),
    }


@pytest.fixture(scope="module")
def lists():
    return words.load(None, words.ANSWERS), words.load(None, words.GUESSES)


@pytest.fixture(scope="module")
def small(lists):
    rng = random.Random(7)
    answers = rng.sample(lists[0], 200)
    guesses = rng.sample(lists[1], 150)
    return Engine(answers, guesses)


def code(colors: str) -> int:
    return sum("-yg".index(c) * 3 ** i for i, c in enumerate(colors))


@pytest.mark.parametrize(
    "guess, answer, colors",
    [
        ("speed", "abide", "--y-y"),
        ("eerie", "there", "y-y-g"),
        ("llama", "hello", "yy---"),
        ("hello", "llama", "--yy-"),
        ("crane", "crane", "ggggg"),
        ("sassy", "asses", "yygy-"),
    ],
)
def test_feedback_duplicates(guess, answer, colors):
    enc = words.encode([guess, answer])
    assert kernels.feedback(enc[0], enc[1]) == code(colors)
    assert reference_feedback(guess, answer) == code(colors)


def test_pattern_matrix_matches_reference(lists):
    rng = random.Random(1)
    guesses = rng.sample(lists[1], 300)
    answers = rng.sample(lists[0], 300)
    P = kernels.pattern_matrix(words.encode(guesses), words.encode(answers))
    for i, g in enumerate(guesses):
        for j, a in enumerate(answers):
            assert P[i, j] == reference_feedback(g, a), (g, a)


def test_score_matches_reference(small):
    rng = random.Random(2)
    for _ in range(50):
        a, b = rng.sample(small.guesses, 2)
        got = small.score(a, b)
        want = reference_stats(a, b, small.answers)
        assert got.entropy == pytest.approx(want["entropy"])
        assert got.expected == pytest.approx(want["expected"])
        assert (got.outcomes, got.worst, got.singles) == (
            want["outcomes"], want["worst"], want["singles"]
        )


@pytest.fixture(scope="module")
def brute_force(small):
    """Reference stats for every pair of the small engine's guesses."""
    table = {(g, a): reference_feedback(g, a) for g in small.guesses for a in small.answers}
    fb = lambda g, a: table[g, a]  # noqa: E731
    gs = small.guesses
    return [reference_stats(a, b, small.answers, fb) for i, a in enumerate(gs) for b in gs[i + 1:]]


@pytest.mark.parametrize(
    "metric, value, best",
    [("entropy", "entropy", max), ("outcomes", "outcomes", max),
     ("expected", "expected", min), ("worst", "worst", min)],
)
def test_search_finds_true_optimum(small, brute_force, metric, value, best):
    target = best(r[value] for r in brute_force)
    result = small.search([metric], top=5)
    assert getattr(result.boards[metric][0], value) == pytest.approx(target)
    values = [getattr(s, value) for s in result.boards[metric]]
    assert values == sorted(values, reverse=best is max)


@pytest.mark.parametrize("metric", ["entropy", "expected"])
def test_pruning_is_exact(lists, metric):
    rng = random.Random(3)
    engine = Engine(rng.sample(lists[0], 500), rng.sample(lists[1], 1500))
    pruned = engine.search([metric], top=15)
    full = engine.search([metric], top=15, prune=False)
    assert pruned.boards == full.boards
    assert pruned.pairs_scored < full.pairs_scored == full.pairs_total


def test_partner_search_agrees_with_full_search(small):
    best = small.search(["entropy"], top=1).boards["entropy"][0]
    first, second = best.words
    partners = small.search(["entropy"], top=1, partner=first).boards["entropy"]
    assert partners[0].words in [(first, second), (second, first)]


def test_pool_restricts_openers(small):
    result = small.search(["entropy"], top=20, pool="answers")
    answers = set(small.answers)
    assert all(set(s.words) <= answers for s in result.boards["entropy"])


def test_evaluate_plays_every_answer(small):
    answer = small.answers[0]
    other = next(g for g in small.guesses if g not in small.answers)
    ev = small.evaluate(answer, other)
    assert sum(ev.histogram.values()) == small.n
    assert ev.histogram[1] == 1  # the first opener itself
    assert min(ev.histogram) == 1 and ev.mean > 2


def test_cli_json(capsys):
    assert main(["-n", "3", "--pool", "answers", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    board = out["boards"]["entropy"]
    assert len(board) == 3
    assert board[0]["entropy"] >= board[1]["entropy"] >= board[2]["entropy"]


def test_cli_rejects_unknown_word(capsys):
    assert main(["score", "crane", "zzzzz"]) == 2
    assert "not in the guess list" in capsys.readouterr().err
