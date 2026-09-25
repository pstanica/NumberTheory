#!/usr/bin/env python3
"""
Toy implementation of the modified Harvey--Hittmeir search with
quadratic-image (quarter) sieving, following Section 3 of
"Improved deterministic integer factorization by quadratic-image sieving".

Purpose: test CORRECTNESS on small semiprimes (roughly 40-72 bits), not speed.
At these sizes the (1+2t)Y inflation of the regular branch outweighs the
sieve factor rho; the saving is only asymptotic.

What the script does for each random balanced semiprime N = p*q:
  1. m = odd primorial (primes up to --B), S = large prime divisors of m with
     ell > 2t (Lemma 3.8), rho, M, Y = max(2,(t*rho)^(-1/2)) unless --Y is given.
  2. beta with ord_N(beta^{m^2}) > L_*: stand-in for Harvey--Hittmeir
     arXiv:2601.11131 (exact-mod-N BSGS on beta = 2,3,5,...).
  3. Lemma 3.11: certify ord_p(alpha), ord_q(alpha) > L_* (or find a factor).
  4. Loop over candidates (sigma mod m, sigma0 grid).  For each one:
     Gauss-reduce T(L_sigma) (Lemmas 3.1, 3.3); regular/skinny split (Prop 3.9);
     in the regular branch pick k < 2t (Lemma 3.8), compute the local
     translations c_ell (Lemma 3.4 + Prop 3.6), CRT them, form giantstep.
  5. Collision search (Lemma 3.10): exact-match removal plus Lemma 2.4 recovery,
     then product tree + multipoint evaluation over Z/NZ and gcds.
  6. ORACLE CHECKS (use the known p, only for diagnostics): for the correct
     candidate, verify i == Q(k_p) mod ell, that i+c lies in E_R, and that
     alpha^{e*} == giantstep (mod p).

Requires: python-flint (pip3 install python-flint) for fast polynomial
arithmetic modulo a composite N; falls back to a slow pure-Python path.

Usage examples:

  python3 toy_qis_factor.py                       # 5 trials, 56-bit N
  python3 toy_qis_factor.py --bits 64 --trials 10 --seed 7
  python3 toy_qis_factor.py --bits 48 --B 11 --Y 2.0

1. Many trials at 48 bits (about 1 minute)

bash
python3 toy_qis_factor.py --bits 48 --B 7 --trials 300 --seed 101 > run48.log
tail -1 run48.log                                   # "k/300 factored."
grep "oracle" run48.log | grep -c "False"           # failed paper checks (want 0)
grep -o "branch=[a-z]*" run48.log | sort | uniq -c  # regular vs skinny count

2. Several values of Y, at 56 bits

bash
for Y in 1.5 2 3 5 8; do
  python3 toy_qis_factor.py --bits 56 --B 11 --trials 100 --seed 202 --Y $Y > runY$Y.log
  echo "Y=$Y: $(tail -1 runY$Y.log)  failed checks: $(grep oracle runY$Y.log | grep -c False)  $(grep -o 'branch=[a-z]*' runY$Y.log | sort | uniq -c | tr '\n' ' ')"
done

Larger Y sends more candidates to the regular branch, so the sieve gets tested more often.

3. Larger N (about 2 seconds per trial)

bash
python3 toy_qis_factor.py --bits 64 --B 11 --trials 100 --seed 303 > run64.log
python3 toy_qis_factor.py --bits 72 --B 13 --trials 30  --seed 404 > run72.log
for f in run64.log run72.log; do echo "$f: $(tail -1 $f)  failed checks: $(grep oracle $f | grep -c False)"; done

At 72 bits with B = 13 there are three selected primes (t = 3) instead of two.

4. Many seeds at once

bash
for s in $(seq 1 20); do
  python3 toy_qis_factor.py --bits 52 --B 11 --trials 25 --seed $s | tail -1
done | awk '{split($1,a,"/"); ok+=a[1]; n+=a[2]} END {print ok "/" n " factored"}'

5. Print the details of any failure

bash
grep -B6 -E "FAILED|False" run*.log

"""
import argparse, math, random, sys, time
from fractions import Fraction

try:
    import flint
    HAVE_FLINT = True
except ImportError:
    HAVE_FLINT = False

# --------------------------------------------------------------------------
# small number theory helpers
# --------------------------------------------------------------------------
def is_prime(n):
    if n < 2: return False
    for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % sp == 0: return n == sp
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1: break
        else:
            return False
    return True

def rand_prime(lo, hi, rng):
    while True:
        x = rng.randrange(lo, hi) | 1
        if is_prime(x): return x

def primes_upto(B):
    return [x for x in range(3, B + 1) if is_prime(x)]

def chi(x, l):
    x %= l
    if x == 0: return 0
    return 1 if pow(x, (l - 1) // 2, l) == 1 else -1

def crt(residues, moduli):
    x, M = 0, 1
    for r, l in zip(residues, moduli):
        t = ((r - x) * pow(M, -1, l)) % l
        x += M * t
        M *= l
    return x % M

def isqrt_exact(n):
    if n < 0: return None
    r = math.isqrt(n)
    return r if r * r == n else None

# --------------------------------------------------------------------------
# polynomial collision machinery over Z/NZ  (Lemmas 3.9, 3.10)
# --------------------------------------------------------------------------
def poly_from_roots(roots, N):
    """prod (X - r) mod N via a product tree."""
    if HAVE_FLINT:
        ctx = flint.fmpz_mod_poly_ctx(N)
        level = [ctx([-r % N, 1]) for r in roots] or [ctx([1])]
        while len(level) > 1:
            level = [level[i] * level[i + 1] if i + 1 < len(level) else level[i]
                     for i in range(0, len(level), 2)]
        return level[0]
    return list(roots)            # pure-python fallback: keep the roots

def multipoint_eval(f, points, N):
    if HAVE_FLINT:
        return [int(v) for v in f.multipoint_evaluate(points)] if points else []
    out = []                       # O(n*s) fallback
    for x in points:
        v = 1
        for r in f: v = v * (x - r) % N
        out.append(v)
    return out

def collision_gcd(base_vals, giant_vals, N):
    """Find a proper factor of N from a collision mod p between the two lists
    (exact mod-N matches must already be removed).  Returns factor or None."""
    if not base_vals or not giant_vals: return None
    f = poly_from_roots(giant_vals, N)
    vals = multipoint_eval(f, base_vals, N)
    for x, v in zip(base_vals, vals):
        g = math.gcd(v, N)
        if 1 < g < N: return g
        if g == N:                                   # scan once (Lemma 3.10)
            for h in giant_vals:
                g2 = math.gcd(x - h, N)
                if 1 < g2 < N: return g2
    return None

# --------------------------------------------------------------------------
# Lemma 2.4 of HH22: recover p,q from u = a q + b p
# --------------------------------------------------------------------------
def recover(a, b, u, N):
    if b == 0:
        if a != 0 and u % a == 0:
            q = u // a
            if 1 < q < N and N % q == 0: return q
        return None
    if a == 0:
        if u % b == 0:
            p = u // b
            if 1 < p < N and N % p == 0: return p
        return None
    s = isqrt_exact(u * u - 4 * a * b * N)          # b p^2 - u p + a N = 0
    if s is None: return None
    for num in (u + s, u - s):
        if num % (2 * b) == 0:
            p = num // (2 * b)
            if 1 < abs(p) < N and N % abs(p) == 0: return abs(p)
    return None

# --------------------------------------------------------------------------
# large-order stand-in and Lemma 3.11 certification
# --------------------------------------------------------------------------
def order_exceeds_mod_N(alpha, L, N):
    """Exact BSGS modulo N: True iff ord_N(alpha) > L."""
    r = math.isqrt(L) + 1
    table, x = {}, 1
    for i in range(r):
        if x in table: return False
        table[x] = i; x = x * alpha % N
    step = pow(alpha, r, N); y = step
    for j in range(1, L // r + 2):
        if y in table and j * r - table[y] <= L: return False
        y = y * step % N
    return True

def certify_factor_orders(alpha, L, N):
    """Lemma 3.11: returns ('factor', g) or ('ok', None)."""
    r = math.isqrt(L - 1) + 1
    s = L // r
    baby = [pow(alpha, i, N) for i in range(r)]
    step = pow(alpha, r, N)
    giant, y = [], step
    for _ in range(s):
        giant.append(y); y = y * step % N
    g = collision_gcd(baby, giant, N)
    if g: return ('factor', g)
    for d in range(s * r + 1, L + 1):
        g = math.gcd(pow(alpha, d, N) - 1, N)
        if 1 < g < N: return ('factor', g)
    return ('ok', None)

# --------------------------------------------------------------------------
# lattice pieces (Lemmas 3.1, 3.3, 3.8, Prop 3.9)
# --------------------------------------------------------------------------
def gauss_reduce(z1, w1, z2, w2):
    n = lambda z: z[0] * z[0] + z[1] * z[1]
    dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
    if n(z1) > n(z2): z1, w1, z2, w2 = z2, w2, z1, w1
    while True:
        n1 = n(z1)
        mu = (2 * dot(z1, z2) + n1) // (2 * n1)       # round(dot/n1)
        z2 = (z2[0] - mu * z1[0], z2[1] - mu * z1[1])
        w2 = (w2[0] - mu * w1[0], w2[1] - mu * w1[1])
        if n(z2) >= n1: return z1, w1, z2, w2
        z1, w1, z2, w2 = z2, w2, z1, w1

# --------------------------------------------------------------------------
# main experiment
# --------------------------------------------------------------------------
def run_one(N, p, q, args, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    B = args.B
    ells = primes_upto(B)
    m = math.prod(ells)
    g = math.gcd(N, m)
    if 1 < g < N: return {'factor': g, 'how': 'gcd(N,m)'}

    X_target = N ** 0.2
    m0 = max(1, round(X_target / m))
    # selected primes: largest t primes of m with ell > 2t
    S = []
    for t in range(len(ells), 0, -1):
        cand = [l for l in ells if l > 2 * t]
        if len(cand) >= t: S = cand[-t:]; break
    t = len(S)
    F = {l: {x for x in range(1, l) if chi(x, l) == -1 and chi(x - 1, l) == 1} for l in S}
    for l in S: assert 4 * len(F[l]) == l - chi(-1, l)          # Prop 3.6 size
    rho = math.prod(1 - len(F[l]) / l for l in S)
    M = math.prod(S)
    # NB: ||z1||*||z2|| >= D_L forces ||z2|| >= sqrt(D_L), so Y must exceed 1 for
    # the regular branch to occur; at toy sizes (t*rho)^(-1/2) is ~1, so use >= 2.
    Y = args.Y if args.Y else max(2.0, (t * rho) ** -0.5)

    base = math.sqrt(N * m * m0) / (m0 ** 2 * m ** 2)          # = lambda0
    LamR = math.ceil(math.sqrt(2) * (1 + 2 * t) * Y * base) + 2
    LamS = math.ceil(math.sqrt(2) * 2 / (math.sqrt(3) * Y) * base) + 2
    LamHH = math.ceil(math.sqrt(2) * math.sqrt(2 / math.sqrt(3)) * base) + 2  # shortest vector only
    Lreg = 2 * LamR + M
    Lstar = max(Lreg, 2 * LamS) + 1

    log(f"  m={m} (primes<= {B}), m0={m0}, X=m*m0={m*m0}, S={S}, t={t}, M={M}, "
        f"rho={rho:.3f}, Y={Y:.3f}")
    log(f"  lambda0~{base:.1f}, Lambda_R={LamR}, Lambda_S={LamS}, L*={Lstar}")

    # --- large-order element (stand-in) and certification -----------------
    beta = None
    for b0 in (2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21):
        if math.gcd(b0, N) != 1: return {'factor': math.gcd(b0, N), 'how': 'gcd(beta,N)'}
        if order_exceeds_mod_N(pow(b0, m * m, N), Lstar, N): beta = b0; break
    if beta is None: return {'factor': None, 'how': 'no large-order beta found'}
    alpha = pow(beta, m * m, N)
    st, gf = certify_factor_orders(alpha, Lstar, N)
    if st == 'factor': return {'factor': gf, 'how': 'Lemma 3.11 certification'}

    # --- sparse babystep set E_R and dense skinny set ----------------------
    allowed = {l: [r not in F[l] for r in range(l)] for l in S}
    E_R = [e for e in range(Lreg + 1) if all(allowed[l][e % l] for l in S)]
    # powers of alpha (toy: straightforward; paper uses the r + jM scheme)
    apow = [1] * (Lstar + 1)
    for e in range(1, Lstar + 1): apow[e] = apow[e - 1] * alpha % N

    # --- candidate loop ------------------------------------------------------
    sq = math.isqrt(N)
    s0 = max(2, sq // args.kappa)
    grid = []
    while s0 <= sq:
        grid.append(s0); s0 += -(-s0 // m0)             # p in [s0, s0+ceil(s0/m0))
    sbar, s0bar = p % m, max(x for x in grid if x <= p)

    reg, skin = [], []        # giantsteps: (value, a, b, j, c)
    oracle = None
    units = [s for s in range(1, m) if math.gcd(s, m) == 1]
    for sig in units:
        eta = N * pow(sig, -2, m) % m
        for sg0 in grid:
            T = lambda a, b: (N * a, -N * m0 * a + m0 * sg0 * sg0 * b)
            w1, w2 = (1, eta), (0, m)
            z1, w1, z2, w2 = gauss_reduce(T(*w1), w1, T(*w2), w2)
            DL = N * m * m0 * sg0 * sg0
            regular = (z2[0] ** 2 + z2[1] ** 2) <= Y * Y * DL
            if regular:
                for k in range(2 * t):
                    a, b = w1[0] + k * w2[0], w1[1] + k * w2[1]
                    if all(a % l for l in S): break
                else:
                    raise RuntimeError("Lemma 3.8 failed")
                Lam = LamR
            else:
                a, b = w1; Lam = LamS
            m2 = m * m
            tau = (a * N * pow(sig, -1, m2) + b * sig) % m2
            center = Fraction(a * N, sg0) + b * sg0
            tau0 = math.floor(center / m2)
            j = m2 * (tau0 - Lam) + tau
            v = pow(beta, a * N + b - j, N)
            c = 0
            if regular:
                cs = []
                for l in S:
                    mod = m2 * l
                    C = (a * N * pow(sig, -1, mod) + b * sig) % mod
                    R = ((C - tau) % mod) // m2 % l
                    Hn = (-a * N * pow(sig, -1, m * l) + b * sig) % (m * l)
                    H = (Hn // m) % l
                    A = a * N * pow(sig, -3, l) % l
                    Bq = H * pow(sig, -1, l) % l
                    C0 = (R - tau0 + Lam) % l
                    u = (C0 - Bq * Bq * pow(4 * A, -1, l)) % l
                    cs.append((-u) % l if chi(A, l) == 1 else (1 - u) % l)
                c = crt(cs, S)
                v = v * apow[c] % N
                reg.append((v, a, b, j, c))
            else:
                skin.append((v, a, b, j, 0))

            if sig == sbar and sg0 == s0bar:           # ----- ORACLE -----
                u_true = a * q + b * p
                i = (u_true - j) // m2
                assert (u_true - j) % m2 == 0
                kp = (p - sig) // m
                okQ = True
                if regular:
                    for l in S:
                        mod = m2 * l
                        C = (a * N * pow(sig, -1, mod) + b * sig) % mod
                        R = ((C - tau) % mod) // m2 % l
                        Hn = (-a * N * pow(sig, -1, m * l) + b * sig) % (m * l)
                        H = (Hn // m) % l
                        A = a * N * pow(sig, -3, l) % l
                        Bq = H * pow(sig, -1, l) % l
                        C0 = (R - tau0 + Lam) % l
                        okQ &= (A * kp * kp + Bq * kp + C0 - i) % l == 0
                estar = i + c
                oracle = dict(branch='regular' if regular else 'skinny',
                              i=i, in_range=0 <= i <= 2 * Lam, quad_ok=okQ,
                              e_star=estar,
                              in_ER=(estar in set(E_R)) if regular else None,
                              fermat_ok=(pow(alpha, estar, p) == v % p))

    # --- collision search ---------------------------------------------------
    def search(giants, exps):
        table = {apow[e]: e for e in exps}
        rest = []
        for (v, a, b, j, c) in giants:
            if v in table:                                  # exact match
                e = table[v]
                f = recover(a, b, (e - c) * m * m + j, N)
                if f: return f, 'exact match + Lemma 2.4'
                continue                                     # remove it
            rest.append(v)
        g = collision_gcd([apow[e] for e in exps], rest, N)
        return (g, 'collision mod p (multipoint gcd)') if g else (None, None)

    t0 = time.time()
    f, how = search(reg, E_R)
    if not f:
        f, how = search(skin, range(2 * LamS + 1))
    res = dict(factor=f, how=how, oracle=oracle, n_reg=len(reg), n_skin=len(skin),
               E_R=len(E_R), dense_R=Lreg + 1, skinny_babies=2 * LamS + 1,
               hh_babies=2 * LamHH + 1, rho=rho, t=t, search_time=time.time() - t0)
    return res

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bits', type=int, default=56, help='bit size of N (balanced p,q)')
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--B', type=int, default=11, help='primorial bound for m')
    ap.add_argument('--Y', type=float, default=None, help='branch threshold (default max(2,(t rho)^-1/2))')
    ap.add_argument('--kappa', type=int, default=4, help='search p in [sqrt(N)/kappa, sqrt(N)]')
    args = ap.parse_args()
    if not HAVE_FLINT:
        print("python-flint not found: using slow O(n*s) collision fallback "
              "(pip install python-flint)")
    rng = random.Random(args.seed)
    ok = 0
    for tr in range(args.trials):
        h = args.bits // 2
        lo = max(2 ** (h - 1), math.isqrt(2 ** args.bits) // args.kappa)
        while True:
            p = rand_prime(lo, 2 ** h, rng); q = rand_prime(p + 2, 2 ** (h + 1), rng)
            N = p * q
            if p * args.kappa > math.isqrt(N) and math.gcd(N, math.prod(primes_upto(args.B))) == 1: break
        print(f"\nTrial {tr+1}: N={N} ({N.bit_length()} bits), p={p}, q={q}")
        t0 = time.time()
        r = run_one(N, p, q, args)
        good = r['factor'] and N % r['factor'] == 0 and 1 < r['factor'] < N
        ok += bool(good)
        print(f"  RESULT: {'factor ' + str(r['factor']) if good else 'FAILED'}  via {r['how']}  "
              f"({time.time()-t0:.1f}s)")
        if 'oracle' in r and r['oracle']:
            o = r['oracle']
            print(f"  oracle (correct candidate): branch={o['branch']}, i in [0,2Lambda]={o['in_range']}, "
                  f"i==Q(k_p) mod ell={o['quad_ok']}, e* in E_R={o['in_ER']}, alpha^e*==giant mod p={o['fermat_ok']}")
            print(f"  giantsteps: regular={r['n_reg']}, skinny={r['n_skin']}")
            print(f"  babysteps: sieved |E_R|={r['E_R']} of dense {r['dense_R']} "
                  f"(ratio {r['E_R']/r['dense_R']:.3f}, rho={r['rho']:.3f}); skinny {r['skinny_babies']}; "
                  f"plain HH shortest-vector baseline {r['hh_babies']}")
    print(f"\n{ok}/{args.trials} factored.")

if __name__ == '__main__':
    main()
