"""
test_jax_solver.py — numerical regression test for JaxTDGLSolver
=================================================================
Runs N_STEPS of both the original TDGLSolver and the JAX replacement
on a small square film and asserts they stay within tolerance.

Place this file at:  py-tdgl/tdgl/test/test_jax_solver.py

Run:
    python tdgl/test/test_jax_solver.py
    pytest tdgl/test/test_jax_solver.py -v
"""

import tempfile
import numpy as np
import pytest

import tdgl
from tdgl.geometry import box
from tdgl.solver.solver import TDGLSolver
from tdgl.solver.runner import RunningState

# ── parameters ────────────────────────────────────────────────────────────────
MESH_EDGE = 0.6
FILM_SIZE = 8.0
N_STEPS   = 30
DT_INIT   = 0.05
ATOL      = 1e-5
RTOL      = 1e-5


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_device() -> tdgl.Device:
    layer  = tdgl.Layer(coherence_length=1.0, london_lambda=2.0, thickness=0.1)
    film   = tdgl.Polygon("film", points=box(FILM_SIZE, FILM_SIZE))
    device = tdgl.Device(
        "square", layer=layer, film=film,
        terminals=None, probe_points=None, length_units="um",
    )
    device.make_mesh(max_edge_length=MESH_EDGE)
    return device


def make_options(output_file: str) -> tdgl.SolverOptions:
    return tdgl.SolverOptions(
        solve_time=N_STEPS * DT_INIT,   # 0.8.x uses solve_time, not total_time
        dt_init=DT_INIT,
        dt_max=DT_INIT,
        adaptive=False,
        save_every=N_STEPS + 1,
        output_file=output_file,
    )


# ── shared step loop (used by both solvers) ───────────────────────────────────

def step_solver(solver, n_steps: int):
    """Manually drive solver.update() for n_steps, return final (psi, mu)."""
    psi            = solver.psi_init.copy()
    mu             = solver.mu_init.copy()
    supercurrent   = np.zeros(solver.num_edges)
    normal_current = np.zeros(solver.num_edges)
    A_induced      = np.zeros((solver.num_edges, 2))
    dt             = solver.tentative_dt

    # RunningState(names_and_sizes, buffer_size)
    running_state = RunningState({"dt": 1}, n_steps)

    for step in range(n_steps):
        state  = {"step": step, "time": step * dt}
        result = solver.update(
            state, running_state, dt,
            psi=psi, mu=mu,
            supercurrent=supercurrent,
            normal_current=normal_current,
            induced_vector_potential=A_induced,
        )
        dt, psi, mu, supercurrent, normal_current, A_induced = (
            result.dt, result.psi, result.mu,
            result.supercurrent, result.normal_current, result.A_induced,
        )

    return np.asarray(psi), np.asarray(mu)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_jax_available():
    import jax, jax.numpy as jnp
    x = jnp.ones(4)
    assert x.shape == (4,)
    print(f"JAX version: {jax.__version__}  devices: {jax.devices()}")


def test_supercurrent_agree():
    """get_supercurrent_jax must match MeshOperators.get_supercurrent exactly."""
    import jax.numpy as jnp
    from tdgl.solver.jax_solver import JaxOperators

    device = make_device()
    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        solver  = TDGLSolver(device, make_options(f.name))
        jax_ops = JaxOperators(solver.operators)

        psi      = solver.psi_init
        sc_scipy = np.array(solver.operators.get_supercurrent(psi))
        sc_jax   = np.array(jax_ops.get_supercurrent_jax(jnp.array(psi)))

    diff = np.abs(sc_scipy - sc_jax).max()
    print(f"supercurrent max diff: {diff:.2e}")
    assert np.allclose(sc_scipy, sc_jax, atol=1e-10), f"max diff = {diff:.2e}"
    print("PASS")


def test_psi_mu_agree():
    """ψ and μ must match original within tolerance after N_STEPS."""
    from tdgl.solver.jax_solver import JaxTDGLSolver

    device = make_device()
    print(f"\nMesh: {len(device.mesh.sites)} sites, {len(device.mesh.edge_mesh.edges)} edges")

    with tempfile.NamedTemporaryFile(suffix=".h5") as f_orig, \
         tempfile.NamedTemporaryFile(suffix=".h5") as f_jax:

        psi_orig, mu_orig = step_solver(
            TDGLSolver(device, make_options(f_orig.name)), N_STEPS
        )
        psi_jax, mu_jax = step_solver(
            JaxTDGLSolver(device, make_options(f_jax.name)), N_STEPS
        )

    abs_orig = np.abs(psi_orig)
    abs_jax  = np.abs(psi_jax)

    psi_diff = np.abs(abs_orig - abs_jax).max()
    mu_diff  = np.abs(mu_orig  - mu_jax ).max()
    print(f"|ψ| max diff : {psi_diff:.2e}")
    print(f"μ   max diff : {mu_diff:.2e}")

    assert np.allclose(abs_orig, abs_jax, atol=ATOL, rtol=RTOL), \
        f"|ψ| max diff = {psi_diff:.2e} > {ATOL}"
    assert np.allclose(mu_orig, mu_jax, atol=ATOL, rtol=RTOL), \
        f"μ max diff = {mu_diff:.2e} > {ATOL}"
    print("PASS")


# ── standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("test_jax_available")
    test_jax_available()

    print("\ntest_supercurrent_agree")
    test_supercurrent_agree()

    print("\ntest_psi_mu_agree")
    test_psi_mu_agree()

    print("\nAll tests passed.")
