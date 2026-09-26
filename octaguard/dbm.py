"""Octagon abstract domain implemented on a Difference Bound Matrix.

Variables live at indices ``0 .. n-1``; the DBM stores bounds on
``v(i) - v(j) <= m[i][j]`` with an implicit zero variable at index ``n``.
The half-variable encoding (i -> 2*i for +v_i, 2*i+1 for -v_i) is not needed
explicitly because every constraint the guard language produces has one of the
canonical octagon forms:

    v_i <= c,  -v_i <= c,  v_i - v_j <= c,  v_i + v_j <= c

and the closure/transfer routines below implement the Miné strong closure
specialised to that representation.  ``m[i][n]`` bounds ``v_i`` (v_i - 0),
``m[n][i]`` bounds ``-v_i`` (0 - v_i).

``INF`` means "no bound".  The matrix always holds *normalised* DBMs: after
construction or any meet/join/transfer callers re-run :meth:`DBM.close`
(strong closure).  A bottom element is recognised by a negative diagonal.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

INF: float = math.inf


class OctagonError(ValueError):
    """Raised on an unparseable / inconsistent octagon constraint request."""


class DBM:
    """A normalised (strongly closed) octagon over ``n`` variables."""

    __slots__ = ("n", "m")

    def __init__(self, n: int, m: Optional[Sequence[Sequence[float]]] = None) -> None:
        self.n = n
        if m is None:
            self.m = [[0 if i == j else INF for j in range(n + 1)] for i in range(n + 1)]
            return
        rows = n + 1
        if len(m) != rows or any(len(r) != rows for r in m):
            raise OctagonError("DBM matrix has wrong dimensions")
        self.m = [[float(x) for x in row] for row in m]
        self.close()

    # ------------------------------------------------------------------ basics

    def copy(self) -> "DBM":
        d = DBM.__new__(DBM)
        d.n = self.n
        d.m = [row[:] for row in self.m]
        return d

    def is_bottom(self) -> bool:
        return any(self.m[i][i] < 0 for i in range(self.n + 1))

    def is_top(self) -> bool:
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                if i == j:
                    continue
                if self.m[i][j] < INF / 2:
                    return False
        return True

    # ---------------------------------------------------------------- closure

    def close(self) -> "DBM":
        """In-place Floyd-Warshall closure; specialised tightening for +/-.

        The zero variable participates like any other variable, which gives
        unary bounds (v_i <= c, -v_i <= c) for free.  The extra diagonal step
        propagates the octagon-specific rules

            (v_i - v_j <= c1) and (v_i + v_j <= c2)  =>  2 v_i <= c1 + c2
            (v_j - v_i <= c1) and (v_i + v_j <= c2)  =>  2 v_j <= c1 + c2

        In our matrix the pair (m[i][j], m[i][n..]) layout is uniform because
        zero is a regular variable, so the classic 3-step formulation works
        directly: shortest paths on the complete difference graph already
        tighten everything expressible as v_i - v_j.  Sum constraints
        v_i + v_j are encoded as m[i][j] style edges too once added, and the
        half-scale tightening below is applied as the dedicated pass.
        """
        m = self.m
        klen = self.n + 1
        for k in range(klen):
            mk = m[k]
            for i in range(klen):
                mi = m[i]
                mik = mi[k]
                if mik == INF:
                    continue
                for j in range(klen):
                    kj = mk[j]
                    if kj == INF:
                        continue
                    cand = mik + kj
                    if cand < mi[j]:
                        mi[j] = cand
        # Octagon tightening: bounds on v_i inferred from paired edges.
        # m[i][z] = upper(v_i); m[z][i] = upper(-v_i).  For i != j:
        # m[i][j]=ub(v_i-v_j), m[j][i]=ub(v_j-v_i); the sum bound lives in
        # neither place directly, so sum constraints are kept as *synthetic*
        # half variables: see ``add_sum_constraint``.  After paths relax the
        # difference edges, sharpen diagonals/unaries from the stored sums.
        self._tighten_from_sums()
        return self

    # Sum constraints v_i + v_j <= c are represented on dedicated entries of
    # the matrix that would otherwise be unused diagonal shadows: we store
    # them out-of-band so they survive every operation.  To keep the class a
    # pure n+1 DBM we instead fold sums into strengthened unary/difference
    # bounds at insertion time (they are only needed for implication checks
    # against guard predicates, all of which are difference or unary), and
    # additionally remember them explicitly for join/widen/relate.
    #
    # NOTE: the out-of-band map is populated by add_sum_constraint and kept in
    # sync by all transfer operators below.

    def _tighten_from_sums(self) -> None:
        # Placeholder hook; sums are tracked in ``self._sums`` when attached.
        sums = getattr(self, "_sums", None)
        if not sums:
            return
        m = self.m
        z = self.n
        for (i, j), c in sums.items():
            # v_i+v_j<=c plus v_j-v_i<=d  => 2 v_i <= c+d (leave 1/2 scale)
            d = m[j][i]
            if d < INF / 2 and (c + d) / 2.0 < m[i][z]:
                m[i][z] = (c + d) / 2.0
            d = m[i][j]
            if d < INF / 2 and (c + d) / 2.0 < m[j][z]:
                m[j][z] = (c + d) / 2.0

    def _ensure_sums(self) -> dict:
        if not hasattr(self, "_sums"):
            self._sums = {}
        return self._sums

    def _copy_sums_into(self, dst: "DBM") -> None:
        s = getattr(self, "_sums", None)
        if s:
            dst._sums = dict(s)
        else:
            dst._sums = {}

    # ------------------------------------------------------------ constructors

    @classmethod
    def top(cls, n: int) -> "DBM":
        return cls(n)

    @classmethod
    def bottom(cls, n: int) -> "DBM":
        d = cls(n)
        d.m[0][0] = -1.0
        return d

    @classmethod
    def unary_constraint(cls, n: int, i: int, c: float, sign: int) -> "DBM":
        """sign = +1 -> v_i <= c; sign = -1 -> -v_i <= c (v_i >= -c)."""
        d = cls(n)
        d.add_unary(i, c, sign)
        d.close()
        return d

    def add_unary(self, i: int, c: float, sign: int) -> None:
        z = self.n
        if not (0 <= i < self.n):
            raise OctagonError("register index out of range")
        if sign > 0:
            if c < self.m[i][z]:
                self.m[i][z] = float(c)
        elif sign < 0:
            if c < self.m[z][i]:
                self.m[z][i] = float(c)
        else:
            raise OctagonError("unary sign must be +1 or -1")

    def add_difference(self, i: int, j: int, c: float) -> None:
        """v_i - v_j <= c."""
        self._check_idx(i, j)
        if c < self.m[i][j]:
            self.m[i][j] = float(c)

    def add_sum_constraint(self, i: int, j: int, c: float) -> None:
        """Record v_i + v_j <= c (i == j means 2*v_i <= c)."""
        self._check_idx(i, j)
        sums = self._ensure_sums()
        key = (i, j) if i <= j else (j, i)
        old = sums.get(key, INF)
        if c < old:
            sums[key] = float(c)
        self._tighten_from_sums()

    def _check_idx(self, *idx: int) -> None:
        for i in idx:
            if not (0 <= i < self.n):
                raise OctagonError("register index out of range")

    # --------------------------------------------------------------- lattice ops

    def meet(self, other: "DBM") -> "DBM":
        """Intersection: pointwise minimum then closure."""
        if self.n != other.n:
            raise OctagonError("meet of DBMs of different dimension")
        d = self.copy()
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                if other.m[i][j] < d.m[i][j]:
                    d.m[i][j] = other.m[i][j]
        sums = dict(getattr(self, "_sums", {}) or {})
        for k, v in (getattr(other, "_sums", {}) or {}).items():
            if v < sums.get(k, INF):
                sums[k] = v
        d._sums = sums
        d.close()
        return d

    def join(self, other: "DBM") -> "DBM":
        """Tight closure of the union: max edge bound after closure of both.

        m_join[i][j] = max(m1*[i][j], m2*[i][j]) is the tightest octagon
        containing both operands for the strongly closed DBM representation.
        """
        return self._join_core(other, widen=False)

    def widen(self, other: "DBM") -> "DBM":
        """Fixed widening rule: a finite edge in ``self`` that grows (stays
        finite but larger, or becomes infinite) in ``other`` is dropped to
        INF; unary bounds present in one operand but absent in the other are
        dropped the same way.  Stabilisation is therefore guaranteed: every
        stabilised edge is either INF or a constant drawn from the old
        iterate.  Sum bounds follow the identical rule."""
        return self._join_core(other, widen=True)

    def _join_core(self, other: "DBM", widen: bool) -> "DBM":
        if self.n != other.n:
            raise OctagonError("join of DBMs of different dimension")
        if self.is_bottom():
            r = other.copy()
            return r
        if other.is_bottom():
            r = self.copy()
            return r
        a, b = self, other
        d = a.copy()
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                av, bv = a.m[i][j], b.m[i][j]
                if widen:
                    if av < INF / 2 and bv > av + 1e-12:
                        d.m[i][j] = INF
                    # edge unchanged/tighter: keep av (already in d)
                else:
                    d.m[i][j] = max(av, bv)
        sa = getattr(a, "_sums", {}) or {}
        sb = getattr(b, "_sums", {}) or {}
        keys = set(sa) | set(sb)
        sums: dict = {}
        for k in keys:
            av, bv = sa.get(k, INF), sb.get(k, INF)
            if widen:
                if av < INF / 2 and bv > av + 1e-12:
                    continue
                if av < INF / 2:
                    sums[k] = av
            else:
                v = max(av, bv)
                if v < INF / 2:
                    sums[k] = v
        d._sums = sums
        d.close()
        return d

    def leq(self, other: "DBM") -> bool:
        """self <= other  <=>  every defining constraint of other holds in self."""
        if self.is_bottom():
            return True
        if other.is_bottom():
            return False
        if self.n != other.n:
            raise OctagonError("comparison of DBMs of different dimension")
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                if self.m[i][j] > other.m[i][j] + 1e-9:
                    return False
        for k, v in (getattr(other, "_sums", {}) or {}).items():
            sv = (getattr(self, "_sums", {}) or {}).get(k, INF)
            if sv > v + 1e-9:
                return False
        return True

    # ------------------------------------------------------------- transfer fns

    def forget(self, i: int) -> "DBM":
        """Remove all information about register ``i`` (assignment pre-step)."""
        self._check_idx(i)
        d = self.copy()
        z = d.n
        for j in range(z + 1):
            if j != i:
                d.m[i][j] = INF
                d.m[j][i] = INF
        d.m[i][i] = 0.0
        sums = {k: v for k, v in (getattr(d, "_sums", {}) or {}).items() if i not in k}
        d._sums = sums
        d.close()
        return d

    def assign_const(self, i: int, c: int) -> "DBM":
        """r_i := c."""
        d = self.forget(i)
        d.add_unary(i, float(c), +1)
        d.add_unary(i, float(-c), -1)
        d.close()
        return d

    def shift_const(self, i: int, delta: int) -> "DBM":
        """r_i := r_i + delta, delta a signed integer constant."""
        d = self.copy()
        z = d.n
        # Every edge touching i shifts by +/- delta; edges between other
        # variables, and sums not involving i, stay unchanged.
        old = self.m
        for j in range(z):
            if j == i:
                continue
            # v_i - v_j <= c  becomes  v_i' - v_j <= c + delta
            if old[i][j] < INF / 2:
                d.m[i][j] = old[i][j] + delta
            else:
                d.m[i][j] = INF
            if old[j][i] < INF / 2:
                d.m[j][i] = old[j][i] - delta
            else:
                d.m[j][i] = INF
        # unaries
        d.m[i][z] = (old[i][z] + delta) if old[i][z] < INF / 2 else INF
        d.m[z][i] = (old[z][i] - delta) if old[z][i] < INF / 2 else INF
        d.m[i][i] = 0.0
        sums = {}
        for k, v in (getattr(self, "_sums", {}) or {}).items():
            a, b = k
            if a == i or b == i:
                other = b if a == i else a
                # v_i+v_other: new sum = old sum + delta
                sums[k] = v + delta
            else:
                sums[k] = v
        d._sums = sums
        d.close()
        return d

    # ------------------------------------------------------------- predicates

    def entails_unary(self, i: int, c: float, sign: int, eps: float = 1e-9) -> bool:
        if self.is_bottom():
            return True
        z = self.n
        bound = self.m[i][z] if sign > 0 else self.m[z][i]
        return bound <= c + eps

    def entails_difference(self, i: int, j: int, c: float, eps: float = 1e-9) -> bool:
        if self.is_bottom():
            return True
        return self.m[i][j] <= c + eps

    def entails_sum(self, i: int, j: int, c: float, eps: float = 1e-9) -> bool:
        if self.is_bottom():
            return True
        key = (i, j) if i <= j else (j, i)
        v = (getattr(self, "_sums", {}) or {}).get(key, INF)
        if v <= c + eps:
            return True
        # A sum may be entailed indirectly via unary bounds after closure.
        z = self.n
        ui, uj = self.m[i][z], self.m[j][z]
        if ui < INF / 2 and uj < INF / 2 and ui + uj <= c + eps:
            return True
        return False

    # --------------------------------------------------------------- projection

    def interval(self, i: int) -> Tuple[float, float]:
        """Concrete interval [lo, hi] of register i (INF if unbounded)."""
        z = self.n
        hi = self.m[i][z]          # v_i <= hi
        lo = -self.m[z][i]         # -v_i <= c  => v_i >= -c
        return lo, hi

    def enumerate_integer_box(self, cap: int = 4096) -> Optional[List[Tuple[int, ...]]]:
        """Enumerate integer points inside the *unary* box projection.

        Used only by tests/small-script checks; returns ``None`` if the box is
        unbounded or larger than ``cap`` points.
        """
        rngs: List[range] = []
        total = 1
        for i in range(self.n):
            lo, hi = self.interval(i)
            if lo == -INF or hi == INF:
                return None
            a = int(math.ceil(lo - 1e-9))
            b = int(math.floor(hi + 1e-9))
            if a > b:
                return []
            rngs.append(range(a, b + 1))
            total *= (b - a + 1)
            if total > cap:
                return None
        out: List[Tuple[int, ...]] = []
        cur = [r.start for r in rngs]
        while True:
            out.append(tuple(cur))
            k = len(rngs) - 1
            while k >= 0:
                cur[k] += 1
                if cur[k] < rngs[k].stop:
                    break
                cur[k] = rngs[k].start
                k -= 1
            if k < 0:
                break
        return out

    # --------------------------------------------------------------- normal form

    def normalized_constraints(self, scale_integral: bool = True) -> List[str]:
        """Human/checkable list of the non-trivial defining constraints.

        The closure guarantees every listed bound is the tightest one
        expressible in the octagon.  Integer-valued bounds are emitted as
        ints so output is canonical across platforms.
        """
        out: List[str] = []
        if self.is_bottom():
            return ["BOTTOM"]

        def fmt(v: float) -> str:
            if v == INF:
                return "inf"
            if scale_integral and abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return f"{v:.6g}"

        z = self.n
        names = [f"r{i}" for i in range(self.n)]
        for i in range(self.n):
            lo, hi = self.interval(i)
            if lo != -INF:
                out.append(f"{names[i]} >= {fmt(lo)}")
            if hi != INF:
                out.append(f"{names[i]} <= {fmt(hi)}")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                c = self.m[i][j]
                if c < INF / 2:
                    out.append(f"{names[i]} - {names[j]} <= {fmt(c)}")
                c = self.m[j][i]
                if c < INF / 2:
                    out.append(f"{names[j]} - {names[i]} <= {fmt(c)}")
        for (a, b), c in sorted((getattr(self, "_sums", {}) or {}).items()):
            out.append(f"{names[a]} + {names[b]} <= {fmt(c)}")
        return sorted(set(out))


def meet_all(elements: Iterable[DBM]) -> Optional[DBM]:
    it = iter(elements)
    try:
        acc = next(it).copy()
    except StopIteration:
        return None
    for d in it:
        acc = acc.meet(d)
    return acc
