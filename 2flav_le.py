import sympy as sp
from sympy import I, exp, cos, sin, trigsimp
from sympy import Matrix
from sympy import pi
from sympy import Interval
from sympy import Abs
from sympy import conjugate
from sympy import solveset
from sympy import FiniteSet
from sympy import diff
from sympy import re
from sympy import nsolve
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot
from scipy.integrate import solve_ivp

nat_eV_km = 1.973269804e-10
kmtoeV = 1.0 / nat_eV_km

def PMNS(theta):
    s, c = np.sin(theta), np.cos(theta)
    return np.array([[c, s], [-s, c]], dtype=complex)

def electron_potential_eV(x_km, profile, params):
    if profile == "constant":
        return params["V0"]

    if profile == "linear":
        fraction = np.clip(x_km / params["length_km"], 0.0, 1.0)
        return params["V_start"] + fraction * (
            params["V_end"] - params["V_start"]
        )

    if profile == "exponential":
        # V(x) = V0 * exp(-x / scale)
        return params["V0"] * np.exp(-x_km / params["scale_km"])

    raise ValueError(f"Unknown electron density profile: {profile}")

def density_profile(x_km):
    profile = "constant"
    params = {"V0": 0.0}  

    return electron_potential_eV(x_km, profile, params)

def main():
    print("A keveredési szögeket fokban add meg.")
    theta = np.deg2rad(float(input("theta[fok]: ")))
  
    dm = float(input("Delta m^2 [eV^2]: "))

    
    initial_flavor = input("Kezdeti íz (e, a): ").strip().lower()
    length_km = float(input("Maximális L/E [km/GeV]: "))

    flavors = {"e": 0, "tau' ": 1}
    if initial_flavor not in flavors:
        raise ValueError("A kezdeti íz e, mu vagy tau legyen.")

    U = PMNS(theta)
    mass_squared = np.diag([-dm, dm]).astype(complex)
    initial_index = flavors[initial_flavor]

    V_e0 = density_profile(0.0)

    # Hamiltonian in the flavor basis, in eV
    H0_eV = (
        U @ np.diag([-dm, dm]) @ U.conj().T
        / (2.0 * 1e9)
        + np.diag([V_e0, 0.0])
    )

# Matter-Hamiltonian eigenvalues, in eV
    E_matter = np.linalg.eigvalsh(H0_eV)

# Largest eigenvalue separation, in eV
    delta_E_max = E_matter[-1] - E_matter[0]

# Shortest full oscillation length at x = 0, in km
    vis = (
        2.0 * np.pi * 1e9 / (kmtoeV * abs(dm))
        if delta_E_max > 0
        else np.inf
    )

    # A távolság km-ben szerepel a megoldandó egyenletben.
    # A km -> eV^-1 átváltást a jobb oldal tartalmazza.
    def rhs(x_km, psi):
            V_e = density_profile(x_km)
            H_eV = (
                U @ mass_squared @ U.conj().T / (2.0 * 1e9)
                + np.diag([V_e, 0.0])
            )
            return -1j * kmtoeV * (H_eV @ psi)

    psi0 = np.zeros(2, dtype=complex)
    psi0[initial_index] = 1.0

    q_values = np.linspace(0.0, length_km, 1000)

    solution = solve_ivp(
        rhs,
        (0.0, length_km),
        psi0,
        t_eval=q_values,
        rtol=1e-9,
        atol=1e-11,
    )

    probabilities = np.abs(solution.y) ** 2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, flavor in enumerate(("e", "tau' ")):
        plt.plot(q_values, probabilities[i], label=fr"$P_{{{initial_flavor}\to {flavor}}}$")

    y_min = 0.0
    y_max = 1.0 # or set to max you want
    plt.ylim(y_min, y_max)

    y_min, y_max = 0.0, 1.0  # <-- fixed y range
    ax.set_ylim(y_min, y_max)
    def enforce_ylim(event=None):
        # Re-apply fixed y limits and redraw
        ax.set_ylim(y_min, y_max)
        fig.canvas.draw_idle()

    def enforce_x_left(event=None):
        # Get current span and re-apply with left fixed at 0
        x0, x1 = ax.get_xlim()
        span = x1 - x0
        # prevent nonsense (optional)
        if span <= 0:
            return
        ax.set_xlim(0, 0 + span)
        fig.canvas.draw_idle()

    # Try multiple hooks that get called after interactions
    fig.canvas.mpl_connect('draw_event', enforce_ylim)
    fig.canvas.mpl_connect('button_release_event', enforce_ylim)
    fig.canvas.mpl_connect('scroll_event', enforce_ylim)
    fig.canvas.mpl_connect('resize_event', enforce_ylim)

    fig.canvas.mpl_connect('draw_event', enforce_x_left)
    fig.canvas.mpl_connect('button_release_event', enforce_x_left)
    fig.canvas.mpl_connect('scroll_event',enforce_x_left)
    fig.canvas.mpl_connect('resize_event', enforce_x_left)


    plt.xlim(0, 20*vis)
    plt.xlabel('L/E (km/GeV)')
    plt.ylabel('Probability')   
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()