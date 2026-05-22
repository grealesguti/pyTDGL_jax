"""
jax_solver.py — JAX backend for py-tdgl, Phase 1 + 2
=====================================================
Drop this file into  tdgl/solver/jax_solver.py

Phase 1  Dense ψ update  (solve_for_psi_squared_jax, adaptive_euler_step_jax)
Phase 2  Sparse operators  (JaxOperators wrapping psi_laplacian, divergence, etc.)

Nothing in the original TDGLSolver is modified.  The JAX functions are called
from a thin subclass  JaxTDGLSolver  that overrides only the two hot methods.

Usage
-----
    from tdgl.solver.jax_solver import JaxTDGLSolver
    solution = tdgl.solve(device, options, ...)        # unchanged API
    # — or —
    solver = JaxTDGLSolver(device, options, ...)
    solution = solver.solve()

Phases not yet implemented here
---------------------------------
  Phase 4  jax.lax.scan / while_loop over time steps
  Phase 5  jax.vmap / jax.grad
"""

from __future__ import annotations

import itertools
import logging
from functools import partial
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

# ── JAX imports ──────────────────────────────────────────────────────────────
try:
    import jax
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO
    import klujax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Enable 64-bit precision: required for complex128 psi and float64 mu/klujax
if JAX_AVAILABLE:
    jax.config.update("jax_enable_x64", True)

from .solver import SolverResult, TDGLSolver
from .options import SolverOptions
from ..device.device import Device
from ..solution.solution import Solution

logger = logging.getLogger("jax_solver")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: scipy sparse → JAX BCOO
# ─────────────────────────────────────────────────────────────────────────────

def scipy_to_bcoo(mat: sp.spmatrix) -> "BCOO":
    """Convert any scipy sparse matrix to a JAX BCOO array.

    The indices are frozen (static).  Only .data changes when link
    variables are updated (see JaxOperators.update_link_variables).
    """
    coo = mat.tocoo()
    indices = jnp.array(np.stack([coo.row, coo.col], axis=1), dtype=jnp.int32)
    data    = jnp.array(coo.data)
    return BCOO((data, indices), shape=mat.shape)


def update_bcoo_data(bcoo: "BCOO", new_data: np.ndarray) -> "BCOO":
    """Return a new BCOO with updated .data but the same frozen indices."""
    return BCOO((jnp.array(new_data), bcoo.indices), shape=bcoo.shape)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: JAX sparse operator container
# ─────────────────────────────────────────────────────────────────────────────

class JaxOperators:
    """Thin wrapper around MeshOperators that keeps JAX BCOO mirrors.

    Call  jax_ops = JaxOperators(mesh_operators)  after
    operators.build_operators() + operators.set_link_exponents() have run.

    Attributes
    ----------
    psi_laplacian_bcoo   : BCOO  — covariant Laplacian (complex, updated each step)
    psi_gradient_bcoo    : BCOO  — covariant gradient  (complex, updated each step)
    divergence_bcoo      : BCOO  — divergence operator (real, fixed)
    mu_gradient_bcoo     : BCOO  — μ gradient          (real, fixed)
    mu_boundary_lap_bcoo : BCOO  — Neumann boundary term for μ (real, fixed)

    edges0, edges1       : jnp arrays — edge endpoint indices (for get_supercurrent_jax)
    """

    def __init__(self, mesh_operators):
        assert JAX_AVAILABLE, "JAX is not installed."
        ops = mesh_operators

        # Fixed-topology operators (convert once, never touch again)
        self.divergence_bcoo      = scipy_to_bcoo(ops.divergence)
        self.mu_gradient_bcoo     = scipy_to_bcoo(ops.mu_gradient)
        self.mu_boundary_lap_bcoo = scipy_to_bcoo(ops.mu_boundary_laplacian)

        # Edge endpoint arrays needed for supercurrent calculation
        self.edges0 = jnp.array(ops.edges[:, 0], dtype=jnp.int32)
        self.edges1 = jnp.array(ops.edges[:, 1], dtype=jnp.int32)

        # Store frozen index arrays separately so we can rebuild data cheaply
        # psi_laplacian
        coo_lap = ops.psi_laplacian.tocoo()
        self._lap_indices = jnp.array(
            np.stack([coo_lap.row, coo_lap.col], axis=1), dtype=jnp.int32
        )
        self._lap_shape = ops.psi_laplacian.shape
        self.psi_laplacian_bcoo = BCOO(
            (jnp.array(coo_lap.data), self._lap_indices), shape=self._lap_shape
        )

        # mu_laplacian COO triplets (frozen — topology never changes)
        # Stored as JAX arrays so klujax.solve stays fully inside JAX.
        coo_mu = ops.mu_laplacian.tocoo()
        self._mu_lap_Ai   = jnp.array(coo_mu.row,  dtype=jnp.int32)
        self._mu_lap_Aj   = jnp.array(coo_mu.col,  dtype=jnp.int32)
        self._mu_lap_Ax   = jnp.array(coo_mu.data, dtype=jnp.float64)
        self._mu_lap_shape = ops.mu_laplacian.shape

        # psi_gradient
        coo_grad = ops.psi_gradient.tocoo()
        self._grad_indices = jnp.array(
            np.stack([coo_grad.row, coo_grad.col], axis=1), dtype=jnp.int32
        )
        self._grad_shape = ops.psi_gradient.shape
        self.psi_gradient_bcoo = BCOO(
            (jnp.array(coo_grad.data), self._grad_indices), shape=self._grad_shape
        )

    def sync_from(self, mesh_operators) -> None:
        """Pull updated link-variable data from mesh_operators after
        set_link_exponents() has been called.  Indices never change.
        """
        ops = mesh_operators
        coo_lap  = ops.psi_laplacian.tocoo()
        coo_grad = ops.psi_gradient.tocoo()
        self.psi_laplacian_bcoo = BCOO(
            (jnp.array(coo_lap.data),  self._lap_indices),  shape=self._lap_shape
        )
        self.psi_gradient_bcoo = BCOO(
            (jnp.array(coo_grad.data), self._grad_indices), shape=self._grad_shape
        )

    # ------------------------------------------------------------------
    # JAX equivalents of MeshOperators methods
    # ------------------------------------------------------------------

    def get_supercurrent_jax(self, psi: jnp.ndarray) -> jnp.ndarray:
        """JAX port of MeshOperators.get_supercurrent(psi).

        Original:
            (psi.conjugate()[edges[:, 0]] * (psi_gradient @ psi)).imag
        """
        return (psi.conjugate()[self.edges0] * (self.psi_gradient_bcoo @ psi)).imag


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: pure JAX psi update
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("gamma", "u"))
def solve_for_psi_squared_jax(
    psi:         jnp.ndarray,   # complex128, (N_sites,)
    abs_sq_psi:  jnp.ndarray,   # float64,    (N_sites,)
    mu:          jnp.ndarray,   # float64,    (N_sites,)
    epsilon:     jnp.ndarray,   # float64,    (N_sites,)
    gamma:       float,
    u:           float,
    dt:          float,
    psi_lap:     "BCOO",        # complex128 BCOO, (N_sites, N_sites)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX port of TDGLSolver.solve_for_psi_squared.

    Returns
    -------
    new_psi      : complex128 (N_sites,)
    new_sq_psi   : float64   (N_sites,)
    valid        : bool scalar  — False if discriminant < 0 anywhere
                   (caller should halve dt and retry, same as original)
    """
    U = jnp.exp((-1j * mu * dt).astype(jnp.complex128))
    z = U * (gamma ** 2 / 2.0) * psi

    psi_lap_psi = psi_lap @ psi

    w = z * abs_sq_psi + U * (
        psi
        + (dt / u)
        * jnp.sqrt(1.0 + gamma ** 2 * abs_sq_psi)
        * ((epsilon - abs_sq_psi) * psi + psi_lap_psi)
    )

    c          = w.real * z.real + w.imag * z.imag
    two_c_1    = 2.0 * c + 1.0
    w2         = jnp.abs(w) ** 2
    abs_z_sq   = jnp.abs(z) ** 2
    discriminant = two_c_1 ** 2 - 4.0 * abs_z_sq * w2

    valid = jnp.all(discriminant >= 0.0)

    # Safe sqrt: clamp to 0 so no NaN even on the invalid branch
    safe_disc  = jnp.maximum(discriminant, 0.0)
    new_sq_psi = jnp.where(
        discriminant >= 0.0,
        (2.0 * w2) / (two_c_1 + jnp.sqrt(safe_disc)),
        abs_sq_psi,   # fallback: keep old value on failure sites
    )
    new_psi = jnp.where(discriminant >= 0.0, w - z * new_sq_psi, psi)

    return new_psi, new_sq_psi, valid


def adaptive_euler_step_jax(
    step:        int,
    psi:         jnp.ndarray,
    abs_sq_psi:  jnp.ndarray,
    mu:          jnp.ndarray,
    epsilon:     jnp.ndarray,
    gamma:       float,
    u:           float,
    dt:          float,
    psi_lap:     "BCOO",
    options,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Adaptive Euler step using JAX psi update.

    Mirrors TDGLSolver.adaptive_euler_step: retries with smaller dt
    while discriminant < 0.  dt halving is done in Python (not inside
    jit) because it's rare and affects a scalar — no performance cost.
    """
    new_psi, new_sq_psi, valid = solve_for_psi_squared_jax(
        psi, abs_sq_psi, mu, epsilon, gamma, u, dt, psi_lap
    )

    for retries in itertools.count():
        if bool(valid):
            break
        if not options.adaptive or retries > options.max_solve_retries:
            raise RuntimeError(
                f"JAX solver failed to converge in {options.max_solve_retries}"
                f" retries at step {step} with dt = {dt:.2e}."
                f" Try a smaller dt_init."
            )
        dt = dt * options.adaptive_time_step_multiplier
        new_psi, new_sq_psi, valid = solve_for_psi_squared_jax(
            psi, abs_sq_psi, mu, epsilon, gamma, u, dt, psi_lap
        )

    return new_psi, new_sq_psi, dt


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1+2 combined: JAX solve_for_observables
# ─────────────────────────────────────────────────────────────────────────────

def solve_for_observables_jax(
    psi:          jnp.ndarray,
    dA_dt:        Union[float, jnp.ndarray],
    mu_boundary:  jnp.ndarray,
    jax_ops:      JaxOperators,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX port of TDGLSolver.solve_for_observables.

    Phase 3: fully inside JAX — klujax replaces the SciPy sparse LU solve.
    The entire function is now differentiable via klujax's custom VJP.
    """
    supercurrent = jax_ops.get_supercurrent_jax(psi)

    rhs = (
        jax_ops.divergence_bcoo @ (supercurrent - dA_dt)
        - jax_ops.mu_boundary_lap_bcoo @ mu_boundary
    )
    # Phase 3: klujax direct sparse solve — no NumPy round-trip, AD-compatible
    mu = klujax.solve(
        jax_ops._mu_lap_Ai,
        jax_ops._mu_lap_Aj,
        jax_ops._mu_lap_Ax,
        rhs,
    )

    normal_current = -(jax_ops.mu_gradient_bcoo @ mu) - dA_dt
    return mu, supercurrent, normal_current


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in subclass
# ─────────────────────────────────────────────────────────────────────────────

class JaxTDGLSolver(TDGLSolver):
    """TDGLSolver with Phase 1+2+3 hot methods replaced by JAX equivalents.

    Phase 1: psi update (solve_for_psi_squared_jax)
    Phase 2: sparse operators (BCOO psi_laplacian, divergence, gradients)
    Phase 3: Poisson solve via klujax (differentiable, no SciPy round-trip)

    All initialisation, I/O, screening, and runner logic is inherited
    unchanged from TDGLSolver.

    The first call to update() triggers JIT compilation (~seconds).
    Subsequent steps run the compiled kernels.

    AD is now available end-to-end through the full time step.
    """

    def __init__(self, *args, **kwargs):
        assert JAX_AVAILABLE, (
            "JAX is not installed.  Run: pip install jax jaxlib"
        )
        super().__init__(*args, **kwargs)

        # Build JAX operator mirrors after the parent __init__ has
        # already called build_operators() + set_link_exponents()
        self._jax_ops = JaxOperators(self.operators)

        # Convert initial state arrays to JAX
        self._psi_jax = jnp.array(self.psi_init)
        self._mu_jax  = jnp.array(self.mu_init)
        self._mu_boundary_jax = jnp.array(self.mu_boundary)
        self._epsilon_jax     = jnp.array(self.epsilon)

        logger.info("JaxTDGLSolver initialised — Phase 1+2+3 active (klujax Poisson solve).")

    # ------------------------------------------------------------------
    # Override: adaptive_euler_step
    # ------------------------------------------------------------------

    def adaptive_euler_step(
        self,
        step:       int,
        psi,
        abs_sq_psi,
        mu,
        epsilon,
        dt:         float,
    ):
        """Calls the JAX psi update instead of the NumPy/CuPy one."""
        psi_j        = jnp.array(psi)       if not isinstance(psi, jnp.ndarray)        else psi
        abs_sq_psi_j = jnp.array(abs_sq_psi) if not isinstance(abs_sq_psi, jnp.ndarray) else abs_sq_psi
        mu_j         = jnp.array(mu)        if not isinstance(mu, jnp.ndarray)         else mu
        epsilon_j    = jnp.array(epsilon)   if not isinstance(epsilon, jnp.ndarray)    else epsilon

        # Sync link variables from the scipy operator (set by parent's update())
        self._jax_ops.sync_from(self.operators)

        new_psi, new_sq_psi, dt = adaptive_euler_step_jax(
            step, psi_j, abs_sq_psi_j, mu_j, epsilon_j,
            self.gamma, self.u, dt,
            self._jax_ops.psi_laplacian_bcoo,
            self.options,
        )
        return new_psi, new_sq_psi, dt

    # ------------------------------------------------------------------
    # Override: solve_for_observables
    # ------------------------------------------------------------------

    def solve_for_observables(self, psi, dA_dt):
        """Calls the JAX supercurrent + normal-current; SciPy Poisson solve."""
        psi_j = jnp.array(psi) if not isinstance(psi, jnp.ndarray) else psi

        mu_boundary_j = (
            jnp.array(self.mu_boundary)
            if not isinstance(self.mu_boundary, jnp.ndarray)
            else self.mu_boundary
        )

        dA_dt_j = (
            jnp.array(dA_dt)
            if (isinstance(dA_dt, np.ndarray) and not isinstance(dA_dt, jnp.ndarray))
            else dA_dt
        )

        mu, supercurrent, normal_current = solve_for_observables_jax(
            psi_j, dA_dt_j, mu_boundary_j,
            self._jax_ops,
        )
        return mu, supercurrent, normal_current
