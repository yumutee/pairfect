"""Command-line interface: `pairfect search` and `pairfect score`."""

import argparse
import json
import sys
from dataclasses import asdict

from pairfect import words
from pairfect.engine import METRICS, Engine, Evaluation, PairStats


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--answers", metavar="PATH",
        help="possible answers, one word per line (default: Wordle's original 2,315)",
    )
    common.add_argument(
        "--guesses", metavar="PATH",
        help="allowed guesses (default: the 14,855 words the NYT accepts)",
    )
    common.add_argument(
        "--evaluate", action="store_true",
        help="also play every answer out (the pair, then greedy guesses) and"
        " report the average number of guesses",
    )
    common.add_argument("--json", action="store_true", help="print JSON")
    common.add_argument("--threads", type=_positive, help="worker threads (default: all cores)")

    parser = argparse.ArgumentParser(
        prog="pairfect",
        description="Find the best pair of Wordle opening words by exhaustive search.",
    )
    sub = parser.add_subparsers(dest="command")

    search = sub.add_parser(
        "search", parents=[common], help="rank every pair of openers (the default command)",
        description="Rank every pair of opening words and print the best.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="metrics:\n" + "\n".join(f"  {m.name:9} {m.description}" for m in METRICS.values()),
    )
    search.add_argument(
        "-m", "--metric", default="entropy", choices=[*METRICS, "all"],
        help="what makes a pair best (default: entropy)",
    )
    search.add_argument("-n", "--top", type=_positive, default=10, help="pairs to show (default: 10)")
    search.add_argument(
        "--pool", default="all", choices=["all", "answers"],
        help="words allowed as openers: any allowed guess, or only possible answers",
    )
    search.add_argument(
        "--with", dest="partner", metavar="WORD",
        help="only pairs containing WORD, i.e. the best partner for WORD",
    )
    search.add_argument(
        "--no-prune", dest="prune", action="store_false",
        help="score every pair even when a bound rules it out",
    )

    score = sub.add_parser(
        "score", parents=[common], help="score one pair of openers",
        description="Score one pair of opening words.",
    )
    score.add_argument("first")
    score.add_argument("second")
    return parser


def _row(rank: str, s: PairStats, ev: Evaluation | None) -> str:
    line = (
        f"{rank:>3}  {s.words[0].upper()} {s.words[1].upper()}  {s.entropy:6.3f}"
        f"  {s.outcomes:8,}  {s.win_by_3:7.1%}  {s.expected:8.3f}  {s.worst:5}"
    )
    if ev:
        line += f"  {ev.mean:7.4f}  {ev.failures:5}"
    return line


def _header(evaluate: bool) -> str:
    line = "  #  pair           bits  outcomes  win by 3  avg left  worst"
    if evaluate:
        line += "  guesses  fails"
    return line


def _progress(done: int, total: int) -> None:
    print(f"\r  first word {done:,}/{total:,}", end="", file=sys.stderr, flush=True)


def _as_dict(s: PairStats, ev: Evaluation | None) -> dict:
    d = asdict(s) | {"win_by_3": s.win_by_3}
    if ev:
        d |= {"guesses": {str(k): v for k, v in ev.histogram.items()},
              "mean_guesses": ev.mean, "failures": ev.failures}
    return d


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in {"search", "score", "-h", "--help"}:
        argv = ["search", *argv]
    args = _parser().parse_args(argv)

    if args.threads:
        import numba

        numba.set_num_threads(args.threads)

    engine = Engine(
        words.load(args.answers, words.ANSWERS), words.load(args.guesses, words.GUESSES)
    )

    try:
        if args.command == "score":
            stats = engine.score(args.first, args.second)
            ev = engine.evaluate(*stats.words) if args.evaluate else None
            if args.json:
                print(json.dumps(_as_dict(stats, ev), indent=2))
            else:
                print(_header(args.evaluate))
                print(_row("", stats, ev))
            return 0

        metrics = list(METRICS) if args.metric == "all" else [args.metric]
        progress = _progress if sys.stderr.isatty() and not args.partner else None
        result = engine.search(
            metrics, top=args.top, pool=args.pool, partner=args.partner,
            prune=args.prune, progress=progress,
        )
    except (KeyError, ValueError) as e:
        print(f"pairfect: {e.args[0]}", file=sys.stderr)
        return 2
    if progress:
        print("\r\033[K", end="", file=sys.stderr)

    evals = {
        m: [engine.evaluate(*s.words) if args.evaluate else None for s in board]
        for m, board in result.boards.items()
    }

    if args.json:
        print(json.dumps({
            "answers": engine.n,
            "openers": result.openers,
            "pairs_total": result.pairs_total,
            "pairs_scored": result.pairs_scored,
            "seconds": result.seconds,
            "boards": {
                m: [_as_dict(s, ev) for s, ev in zip(board, evals[m])]
                for m, board in result.boards.items()
            },
        }, indent=2))
        return 0

    for m, board in result.boards.items():
        metric = METRICS[m]
        partner = f" with {args.partner.upper()}" if args.partner else ""
        print(f"Best pairs{partner}, {metric.title} ({m}): {metric.description}\n")
        print(_header(args.evaluate))
        for rank, (s, ev) in enumerate(zip(board, evals[m]), 1):
            print(_row(str(rank), s, ev))
        print()
    skipped = result.pairs_total - result.pairs_scored
    note = f" (the other {skipped:,} provably can't make the top {args.top})" if skipped else ""
    print(
        f"{engine.n:,} possible answers, {result.openers:,} openers:"
        f" scored {result.pairs_scored:,} of {result.pairs_total:,} pairs"
        f" in {result.seconds:.1f}s{note}."
    )
    return 0
