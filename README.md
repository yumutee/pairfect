# pairfect

Finds the best **pair** of Wordle opening words by exhaustive, exact search.
You play both words first, then solve. The search scores all ~110 million
pairs of the 14,855 words Wordle accepts against the 2,315 possible answers.

```console
$ uv run pairfect --evaluate
  #  pair           bits  outcomes  win by 3  avg left  worst  guesses  fails
  1  SOARE CLINT   9.633     1,104    47.7%     4.369     25   3.6082      0
  2  ROAST CLINE   9.615     1,087    47.0%     4.371     25   3.6112      0
  3  RIANT SOCLE   9.614     1,081    46.7%     4.314     25   3.6151      0
  ...
```

## What "optimal" means

After you play both openers, the answers are split into groups. Each group
shares the same pair of colour patterns. A good pair makes many small groups.
You can rank pairs by one of four measures (`--metric`):

| metric     | measures                                                           | best pair     |
|------------|--------------------------------------------------------------------|---------------|
| `entropy`  | expected information from the two feedbacks, in bits (default)     | SOARE + CLINT |
| `outcomes` | distinct feedback combinations; ÷ 2,315 = chance to win by guess 3 | SOARE + CLINT |
| `expected` | average number of answers still possible after both guesses        | CARSE + DOILT |
| `worst`    | largest number of answers that can still be possible               | RIANT + COLES |

SOARE + CLINT is best on two of the four measures. It gives 9.63 bits and 1,104
distinct outcomes, so there's a 47.7% chance of winning on guess 3. With
`--evaluate`, every answer is played out: the pair, then greedy follow-up
guesses. SOARE + CLINT then averages 3.61 guesses, and no answer needs more
than six.

If both openers must be possible answers (`--pool answers`), the best pair is
TRICE + SALON.

## Usage

```console
uv run pairfect                          # top 10 pairs by entropy
uv run pairfect -m all -n 5              # top 5 under every metric
uv run pairfect --with crane             # best partner for CRANE
uv run pairfect --pool answers           # both openers must be possible answers
uv run pairfect score salet courd --evaluate
uv run pairfect --json                   # machine-readable output
```

`--answers` and `--guesses` take your own word lists (one word per line).

## How the search works

- `kernels.pattern_matrix` computes the feedback of every guess against every
  answer. The result is a 14,855 × 2,315 table of 0–242 codes, and duplicate
  letters are handled as Wordle handles them.
- `kernels.pair_stats` scores a first word against many second words at once.
  It sorts the answers by the first word's feedback. Then, for each second
  word, it only refines the groups the first word left with two or more
  answers. It uses a 243-slot tally that fits in L1 cache, and runs in
  parallel with numba.
- **Pruning.** For `entropy` and `expected`, candidate first words are visited
  in order of their single-word entropy. A pair is skipped only when it
  provably can't reach the current top *k*:
    - H (a, b) ≤ H (a) + H (b) (subadditivity), and
    - Σ size² ≥ N² · 2^−H (a, b) (Rényi-2 entropy ≤ Shannon entropy).

  With these bounds, the default search scores 21 M of the 110 M pairs.
  `--no-prune` checks every pair and gives the same results: about 60 s for
  all four metrics.
- The tests compare the kernels with a plain-Python Wordle scorer. They also
  compare search results with brute force, and pruned search with exhaustive
  search.

## Caveats

- **Answers.** The default answers are the 2,315 words Wordle shipped with.
  The NYT has since removed a few and now picks answers editorially, so no
  official list exists. Use `--answers` to try a different list.
- **Blind pair.** The second word is fixed and doesn't depend on the first
  word's feedback. An adaptive second guess does better, but then you don't
  have a pair of starting words.
- **Not hard mode.** Hard mode can reject a fixed second word.
- **Evaluation is greedy.** The `guesses` column uses greedy follow-ups (most
  distinct outcomes, then prefer a possible answer). It shows how well a pair
  plays, but it isn't the exact optimum. The ranking metrics above are exact.
