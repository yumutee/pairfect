"""Numba kernels for Wordle feedback and partition statistics."""

import numpy as np
from numba import njit, prange

N_CODES = 243
ALL_GREEN = 242


@njit(cache=True, inline="always")
def feedback(guess, answer):
    """Feedback code for `guess` against `answer` (uint8 letter arrays).

    Duplicate letters follow Wordle's rules: greens are assigned first, then
    yellows left to right, each consuming one unmatched copy of the letter.
    """
    code = 0
    used = 0  # bitmask of answer positions already matched
    weight = 1
    for i in range(5):
        if guess[i] == answer[i]:
            code += 2 * weight
            used |= 1 << i
        weight *= 3
    weight = 1
    for i in range(5):
        if guess[i] != answer[i]:
            for j in range(5):
                if not (used >> j) & 1 and guess[i] == answer[j]:
                    used |= 1 << j
                    code += weight
                    break
        weight *= 3
    return code


@njit(parallel=True, cache=True)
def pattern_matrix(guesses, answers):
    """P[g, a] = feedback code of guess g against answer a."""
    out = np.empty((guesses.shape[0], answers.shape[0]), np.uint8)
    for g in prange(guesses.shape[0]):
        for a in range(answers.shape[0]):
            out[g, a] = feedback(guesses[g], answers[a])
    return out


@njit(parallel=True, cache=True)
def single_entropy(P, xlog2x):
    """Shannon entropy (bits) of each guess's feedback over the answers."""
    G, N = P.shape
    out = np.empty(G)
    for g in prange(G):
        counts = np.zeros(N_CODES, np.int64)
        for a in range(N):
            counts[P[g, a]] += 1
        acc = 0.0
        for v in range(N_CODES):
            acc += xlog2x[counts[v]]
        out[g] = np.log2(N) - acc / N
    return out


@njit(parallel=True, cache=True)
def pair_stats(P, firsts, lo, hi, xlog2x):
    """Joint-partition statistics for the pairs (firsts[r], b), lo[r] <= b < hi[r].

    Playing both words splits the answers into cells that share the same pair
    of feedback codes. For each pair this returns, in (R, G) arrays:
      xsum     - sum over cells of size*log2(size)
      outcomes - number of cells
      sumsq    - sum over cells of size**2
      worst    - size of the largest cell
      singles  - number of cells of size 1
    Entries outside a row's [lo, hi) range are left with outcomes == 0.
    """
    R = firsts.shape[0]
    G, N = P.shape
    xsum = np.zeros((R, G))
    outcomes = np.zeros((R, G), np.int32)
    sumsq = np.zeros((R, G), np.int64)
    worst = np.zeros((R, G), np.int32)
    singles = np.zeros((R, G), np.int32)
    for r in prange(R):
        pa = P[firsts[r]]
        counts = np.zeros(N_CODES, np.int64)
        for j in range(N):
            counts[pa[j]] += 1

        n_single = 0
        n_shared = 0
        n_members = 0
        for v in range(N_CODES):
            if counts[v] == 1:
                n_single += 1
            elif counts[v] > 1:
                n_shared += 1
                n_members += counts[v]
        bounds = np.empty(n_shared + 1, np.int64)
        fill = np.full(N_CODES, -1, np.int64)
        q = 0
        pos = 0
        for v in range(N_CODES):
            if counts[v] > 1:
                bounds[q] = pos
                fill[v] = pos
                pos += counts[v]
                q += 1
        bounds[n_shared] = pos
        members = np.empty(n_members, np.int64)
        for j in range(N):
            if fill[pa[j]] >= 0:
                members[fill[pa[j]]] = j
                fill[pa[j]] += 1

        tally = np.zeros(N_CODES, np.int32)
        for b in range(lo[r], hi[r]):
            pb = P[b]
            n_cells = n_single
            sq = n_single
            n_one = n_single
            big = 1 if n_single > 0 else 0
            xs = 0.0
            for q in range(n_shared):
                s = bounds[q]
                e = bounds[q + 1]
                for t in range(s, e):
                    tally[pb[members[t]]] += 1
                for t in range(s, e):
                    v = pb[members[t]]
                    c = tally[v]
                    if c:
                        tally[v] = 0
                        n_cells += 1
                        sq += c * c
                        xs += xlog2x[c]
                        if c == 1:
                            n_one += 1
                        if c > big:
                            big = c
            xsum[r, b] = xs
            outcomes[r, b] = n_cells
            sumsq[r, b] = sq
            worst[r, b] = big
            singles[r, b] = n_one
    return xsum, outcomes, sumsq, worst, singles


@njit(cache=True)
def best_follow_up(P, cand, cand_guess):
    """Greedy choice of the next guess once the answer is known to be in `cand`."""
    G = P.shape[0]
    n = cand.shape[0]
    is_cand = np.zeros(G, np.bool_)
    for i in range(n):
        is_cand[cand_guess[i]] = True
    tally = np.zeros(N_CODES, np.int32)
    best = -1
    best_cells = -1
    best_is_cand = False
    best_sq = 0
    for g in range(G):
        pg = P[g]
        for t in range(n):
            tally[pg[cand[t]]] += 1
        cells = 0
        sq = 0
        for t in range(n):
            v = pg[cand[t]]
            c = tally[v]
            if c:
                tally[v] = 0
                cells += 1
                sq += c * c
        if (
                cells > best_cells
                or (cells == best_cells and is_cand[g] and not best_is_cand)
                or (cells == best_cells and is_cand[g] == best_is_cand and sq < best_sq)
        ):
            best = g
            best_cells = cells
            best_is_cand = is_cand[g]
            best_sq = sq
            if cells == n and best_is_cand:
                break  # splits every candidate apart and might win outright
    return best
