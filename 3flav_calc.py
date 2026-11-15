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
from scipy.signal import find_peaks

nat_eV_km = 1.973269804e-10
kmtoeV = 1.0 / nat_eV_km

def PMNS(theta12, theta13, theta23, delta_cp):
    s12, c12 = np.sin(theta12), np.cos(theta12)
    s13, c13 = np.sin(theta13), np.cos(theta13)
    s23, c23 = np.sin(theta23), np.cos(theta23)

    U12 = np.array([[c12, s12, 0.0], [-s12, c12, 0.0], [0.0, 0.0, 1.0]], dtype=complex)
    U13 = np.array([[c13, 0.0, s13 * np.exp(-1j * delta_cp)], [0.0, 1.0, 0.0], [-s13 * np.exp(1j * delta_cp), 0.0, c13]], dtype=complex)
    U23 = np.array([[1.0, 0.0, 0.0], [0.0, c23, s23], [0.0, -s23, c23]], dtype=complex)

    return U23 @ U13 @ U12

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
    theta12 = np.deg2rad(float(input("theta12 [fok]: ")))
    theta13 = np.deg2rad(float(input("theta13 [fok]: ")))
    theta23 = np.deg2rad(float(input("theta23 [fok]: ")))
    delta_cp = np.deg2rad(float(input("delta_CP [fok] (ha nincs, 0): ") or "0"))

    dm21 = float(input("Delta m^2_21 [eV^2]: "))
    dm31 = float(input("Delta m^2_31 [eV^2]: "))

    initial_flavor = input("Kezdeti íz (e, mu vagy tau): ").strip().lower()
    length_km = float(input("Maximális L/E [km/GeV]: "))

    flavors = {"e": 0, "mu": 1, "tau": 2}
    if initial_flavor not in flavors:
        raise ValueError("A kezdeti íz e, mu vagy tau legyen.")

    U = PMNS(theta12, theta13, theta23, delta_cp)
    mass_squared = np.diag([0.0, dm21, dm31]).astype(complex)
    initial_index = flavors[initial_flavor]

    V_e0 = density_profile(0.0)

    # Hamiltonian in the flavor basis, in eV
    H0_eV = (
        U @ np.diag([0.0, dm21, dm31]) @ U.conj().T
        / (2.0*1e9)
        + np.diag([V_e0, 0.0, 0.0])
    )

# Matter-Hamiltonian eigenvalues, in eV
    E_matter = np.linalg.eigvalsh(H0_eV)
    print("Matter-Hamiltonian eigenvalues (eV):", E_matter)
# Largest eigenvalue separation, in eV
    delta_E_max = E_matter[-1] - E_matter[0]

# Shortest full oscillation length at x = 0, in km
    vis = (
        2.0 * np.pi *nat_eV_km / delta_E_max
        if delta_E_max > 0
        else np.inf
    )

    # A távolság km-ben szerepel a megoldandó egyenletben.
    # A km -> eV^-1 átváltást a jobb oldal tartalmazza.
    def rhs(x_km, psi):
        V_e = density_profile(x_km)
        H_eV = (
            U @ mass_squared @ U.conj().T / (2.0 * 1e9)
            + np.diag([V_e, 0.0, 0.0])
        )
        return -1j * kmtoeV * (H_eV @ psi)

    psi0 = np.zeros(3, dtype=complex)
    psi0[initial_index] = 1.0

    x_values = np.linspace(0.0, length_km, int(1000*vis))
    solution = solve_ivp(
        rhs,
        (0.0, length_km),
        psi0,
        t_eval=x_values,
        rtol=1e-9,
        atol=1e-11,
    )

    propabilities = np.abs(solution.y) ** 2

    variables =[5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    

    for variable in variables:
        idx = np.argmin(np.abs(x_values - variable))
        evalue = propabilities[0, idx]
        muvalue = propabilities[1, idx]
        tauvalue = propabilities[2, idx]
        print("Propability of electron flavour at L/E = ", variable,":", evalue )
        print("Propability of muon flavour at L/E = ", variable,":", muvalue )
        print("Propability of tau flavour at L/E = ", variable,":", tauvalue )

    def interesting_points(propabilities, x_values):
        points = {}
        for i, flavor in enumerate(("e", "mu", "tau")):
            maximum_index = np.argmax(propabilities[i])
            maximum_x = x_values[maximum_index]
            maximum_probability = propabilities[i, maximum_index]

            points[flavor] = (maximum_x, maximum_probability)

        return points

    points = interesting_points(propabilities, x_values)

    for flavor in ("e", "mu", "tau"):
        maximum_x, depth = points[flavor]
        print(
            f"{flavor}: oscillation depth = {depth:.6f}, "
            f"maximum at x = {maximum_x:.6f}"
        )

    def peak_spacings(propabilities, x_values):
        results = {}

        for i, flavor in enumerate(("e", "mu", "tau")):
            peak_indices, _ = find_peaks(propabilities[i])

            peak_positions = np.asarray(x_values)[peak_indices]
            spacings = np.diff(peak_positions)
            avgspacing = np.mean(spacings) if len(spacings) > 0 else np.nan

            results[flavor] = {
                "peak_positions": peak_positions,
                "spacings": spacings,
                "avgspacing": avgspacing
            }

        return results

    results = peak_spacings(propabilities, x_values)

    for flavor, data in results.items():
        print(f"{flavor} successive peak spacings:", data["spacings"])
        print(f"{flavor} average peak spacing:", data["avgspacing"])

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, flavor in enumerate(("e", "mu", "tau")):
        plt.plot(x_values, propabilities[i], label=fr"$P_{{{initial_flavor}\to {flavor}}}$")

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