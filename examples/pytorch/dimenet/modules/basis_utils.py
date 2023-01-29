import numpy as np
import sympy as sym
from scipy import special as sp
from scipy.optimize import brentq


def Jn(r, n):
    """
    r: int or list
    n: int or list
    len(r) == len(n)
    return value should be the same shape as the input data
    ===
    example:
        r = n = np.array([1, 2, 3, 4])
        res = [0.3, 0.1, 0.1, 0.1]
    ===
    numerical spherical bessel functions of order n
    """
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)  # the same shape as n


def Jn_zeros(n, k):
    """
    n: int
    k: int
    res: array of shape [n, k]

    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    """
    n: int
    res: array of shape [n,]

    n sympy functions
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols("x")

    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    """
    n: int
    k: int
    res: [n, k]

    n * k sympy functions
    Computes the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    """

    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(
                    normalizer[order][i] * f[order].subs(x, zeros[order, i] * x)
                )
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l, m):
    """
    l: int
    m: int
    res: float
    Computes the constant pre-factor for the spherical harmonic of degree l and order m
    input:
    l: int, l>=0
    m: int, -l<=m<=l
    """
    return (
        (2 * l + 1)
        * np.math.factorial(l - abs(m))
        / (4 * np.pi * np.math.factorial(l + abs(m)))
    ) ** 0.5


def associated_legendre_polynomials(l, zero_m_only=True):
    """
    l: int
    return: l sympy functions
    Computes sympy formulas of the associated legendre polynomials up to order l (excluded).
    """
    z = sym.symbols("z")
    P_l_m = [[0] * (j + 1) for j in range(l)]

    P_l_m[0][0] = 1

    if l > 0:
        P_l_m[1][0] = z

        for j in range(2, l):
            P_l_m[j][0] = sym.simplify(
                ((2 * j - 1) * z * P_l_m[j - 1][0] - (j - 1) * P_l_m[j - 2][0])
                / j
            )

        if not zero_m_only:
            for i in range(1, l):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < l:
                    P_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * P_l_m[i][i]
                    )
                for j in range(i + 2, l):
                    P_l_m[j][i] = sym.simplify(
                        (
                            (2 * j - 1) * z * P_l_m[j - 1][i]
                            - (i + j - 1) * P_l_m[j - 2][i]
                        )
                        / (j - i)
                    )

    return P_l_m


def real_sph_harm(l, zero_m_only=True, spherical_coordinates=True):
    """
    return: a sympy function list of length l, for i-th index of the list, it is also a list of length (2 * i + 1)
    Computes formula strings of the real part of the spherical harmonics up to order l (excluded).
    Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.
    """
    if not zero_m_only:
        S_m = [0]
        C_m = [1]
        for i in range(1, l):
            x = sym.symbols("x")
            y = sym.symbols("y")
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]

    P_l_m = associated_legendre_polynomials(l, zero_m_only)

    if spherical_coordinates:
        theta = sym.symbols("theta")
        z = sym.symbols("z")

        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))

        if not zero_m_only:
            phi = sym.symbols("phi")
            for i in range(len(S_m)):
                S_m[i] = (
                    S_m[i]
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )
            for i in range(len(C_m)):
                C_m[i] = (
                    C_m[i]
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )

    Y_func_l_m = [["0"] * (2 * j + 1) for j in range(l)]

    for i in range(l):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j]
                )
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j]
                )

    return Y_func_l_m
