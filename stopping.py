"""Muon mean mass stopping power S(E; material), MeV cm^2 g^-1.  [Step 1.2]

Pure stdlib, no repo dependencies, so it can be checked in isolation like
verify_arithmetic.py.

MODEL
-----
FACT (PDG, "Passage of particles through matter", ref. [12]): the mean rate
of energy loss of a heavy charged particle is the Bethe equation

    -<dE/dx> = K z^2 (Z/A) (1/beta^2)
               [ 0.5 ln( 2 m_e c^2 beta^2 gamma^2 T_max / I^2 )
                 - beta^2 - delta(beta gamma)/2 ],
    K = 4 pi N_A r_e^2 m_e c^2 = 0.307075 MeV mol^-1 cm^2,
    T_max = 2 m_e c^2 beta^2 gamma^2
            / [ 1 + 2 gamma m_e/m_mu + (m_e/m_mu)^2 ].

FACT: delta is the density-effect correction in Sternheimer's
parameterization with x = log10(beta gamma):
    x < x0  : delta = delta0 * 10^(2(x-x0))        (conductors)
    x0<x<x1 : delta = 2 ln10 x - Cbar + a (x1-x)^k
    x > x1  : delta = 2 ln10 x - Cbar.

PROVENANCE WARNING  -- READ BEFORE PUBLISHING
---------------------------------------------
The six Sternheimer constants per material below are transcribed from the
PDG / Berger-Seltzer atomic and nuclear properties tables.  They must be
re-checked against the primary table before any published number depends on
them.  ``validate()`` gives only an INDIRECT check: the minimum mass stopping
power computed from these constants must reproduce the published minima
(Al 1.615, Cu 1.403, Pb 1.122 MeV cm^2 g^-1) near beta*gamma ~ 3-3.5.
Agreement there is strong evidence but is not verification of the table.

NEGLECTED
---------
* Radiative loss (bremsstrahlung, direct pair production, photonuclear), the
  b(E)E term.  No b value is asserted here.  ``energy_loss.py`` instead
  provides a scale-factor sensitivity scan that converts any omission in S
  into an explicit eps_M error bar.
* Delta-ray escape.  The unrestricted mean loss is used, which is the correct
  quantity for the mean momentum degradation of the muon along the path.
"""

import math

M_E = 0.510998950e-3  # GeV
M_MU = 0.10566  # GeV  -- must match config.M_MU
K_BETHE = 0.307075  # MeV mol^-1 cm^2

# Z, A, I[eV], Cbar, x0, x1, a, k, delta0
STERNHEIMER = {
    "Al": dict(
        Z=13.0,
        A=26.98,
        I=166.0,
        Cbar=4.2395,
        x0=0.1708,
        x1=3.0127,
        a=0.08024,
        k=3.6345,
        d0=0.12,
    ),
    "Cu": dict(
        Z=29.0,
        A=63.55,
        I=322.0,
        Cbar=4.4190,
        x0=-0.0254,
        x1=3.2792,
        a=0.14339,
        k=2.9044,
        d0=0.08,
    ),
    "Pb": dict(
        Z=82.0,
        A=207.2,
        I=823.0,
        Cbar=6.2018,
        x0=0.3776,
        x1=3.8073,
        a=0.09359,
        k=3.1608,
        d0=0.14,
    ),
}

# PDG published minimum mass stopping powers, MeV cm^2 g^-1 (validation only)
PDG_MIN_DEDX = {"Al": 1.615, "Cu": 1.403, "Pb": 1.122}

_LN10 = math.log(10.0)


def _delta(bg, m):
    x = math.log10(bg)
    if x >= m["x1"]:
        return 2.0 * _LN10 * x - m["Cbar"]
    if x >= m["x0"]:
        return 2.0 * _LN10 * x - m["Cbar"] + m["a"] * (m["x1"] - x) ** m["k"]
    return m["d0"] * 10.0 ** (2.0 * (x - m["x0"]))


def dedx(p_gev, material):
    """Mean mass stopping power for a muon of momentum p_gev [GeV/c].

    Returns MeV cm^2 g^-1.
    """
    m = STERNHEIMER[material]
    gamma = math.sqrt(1.0 + (p_gev / M_MU) ** 2)
    beta = p_gev / math.sqrt(p_gev * p_gev + M_MU * M_MU)
    bg = beta * gamma
    r = M_E / M_MU
    t_max = 2.0 * M_E * bg * bg / (1.0 + 2.0 * gamma * r + r * r)  # GeV
    arg = (2.0 * M_E * 1e9 * bg * bg) * (t_max * 1e9) / (m["I"] ** 2)  # eV*eV/eV^2
    return (
        K_BETHE
        * m["Z"]
        / m["A"]
        / (beta * beta)
        * (0.5 * math.log(arg) - beta * beta - 0.5 * _delta(bg, m))
    )


def dedx_of_E(E_gev, material):
    """Same, parameterized by total energy E [GeV]."""
    p = math.sqrt(max(E_gev * E_gev - M_MU * M_MU, 1e-14))
    return dedx(p, material)


def validate():
    """Indirect closure test against the PDG published minima."""
    out = {}
    for name, want in PDG_MIN_DEDX.items():
        best_p, best_s = None, float("inf")
        p = 0.05
        while p < 2.0:
            s = dedx(p, name)
            if s < best_s:
                best_s, best_p = s, p
            p *= 1.0005
        out[name] = (best_s, want, best_s / want - 1.0, best_p / M_MU)
    return out


if __name__ == "__main__":
    print(f"{'mat':>4} {'S_min':>8} {'PDG':>8} {'rel':>9} {'(bg)_min':>9}")
    ok = True
    for n, (got, want, rel, bg) in validate().items():
        ok &= abs(rel) < 0.01
        print(f"{n:>4} {got:8.4f} {want:8.4f} {rel * 100:+8.3f}% {bg:9.3f}")
    print("\nRESULT:", "PASS" if ok else "FAIL (Sternheimer constants suspect)")
    print()
    print(f"{'p(GeV/c)':>9} {'S_Al':>8} {'S_Cu':>8} {'S_Pb':>8}")
    for p in (0.5, 0.75, 1.0, 2.0, 3.5, 6.0):
        print(
            f"{p:9.2f} {dedx(p, 'Al'):8.4f} {dedx(p, 'Cu'):8.4f} {dedx(p, 'Pb'):8.4f}"
        )
