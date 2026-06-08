"""Control-bounded leapfrog A/D converter (numpy + scipy, self-contained).

This is the *control-bounded* view of the leapfrog ADC from Feyling, Malmberg,
Wulff & Ytterdal, "A Unified Analysis of Continuous-Time A/D Converters"
(TechRxiv, 2024), and Malmberg's "Control-Bounded Converters" (ETH, 2020). It
differs from the single-loop :class:`~cicadc.sigma_delta.Leapfrog` modulator:
here every integrator has its *own* 1-bit digital control loop (which is what
guarantees the loop stays bounded), and the input is recovered by a digital
*estimation filter* rather than a sinc decimator.

The analog system is the continuous-time state-space

    x'(t) = A x(t) + B u(t) + Gamma s(t)          (integrator chain)
    s_l[n] = sign( (Gamma_tilde^T x)_l )           (local 1-bit control)

with a leap-frog ``A`` (forward gain ``beta`` on the sub-diagonal, backward gain
``alpha`` on the super-diagonal). The estimate is a non-causal FIR

    u_hat[n] = sum_k hf[k] . s[n-1-k] + sum_k hb[k] . s[n+k]

whose taps come from the steady-state Wiener/Kalman solution: two continuous-time
algebraic Riccati equations give ``Vf, Vb``; the forward/backward state matrices
``Af = expm((A - Vf C C^T/eta2) T)`` and ``Ab = expm(-(A + Vb C C^T/eta2) T)``
and input matrices ``Bf, Bb`` then generate the taps ``-W^T Af^k Bf`` and
``-W^T Ab^k Bb`` with ``W = (Vf + Vb)^{-1} B``.

The implementation has been cross-checked against the ``cbadc`` toolbox (same
analog system and estimator): it reconstructs a mid-band sine to ~56 dB in-band
SNR at OSR ~30 (3rd order), a few dB below ``cbadc``'s mpmath-precision result.

NOTE: this module is a validated reference/prototype. Wiring it into the live
visualiser (the estimator is non-causal, and the modulator is continuous-time)
is a separate, larger change tracked for later.
"""
from __future__ import annotations

from math import comb
from typing import Callable, Dict

import numpy as np
import scipy.linalg


def _g_i(N: int) -> float:
    """Proportionality helper from cbadc's leap-frog parameterisation."""
    return float(np.prod([comb(N, k) for k in range(N + 1)]))


class LeapfrogCBADC:
    """Self-contained control-bounded leap-frog ADC (analog system + estimator).

    Parameters
    ----------
    N:
        System order (number of integrators). Default 3.
    BW:
        Target signal bandwidth, in the same time units as ``Ts`` (Hz if seconds).
    ENOB:
        Target effective number of bits; sets the loop gain ``gamma`` and the
        clock period ``T`` exactly as ``cbadc.synthesis.get_leap_frog``.
    xi:
        Proportionality constant (defaults to 4e-3, cbadc's default).
    """

    def __init__(self, N: int = 3, BW: float = 1.0, ENOB: float = 12.0, xi: float = 4e-3) -> None:
        self.N = N
        self.BW = BW
        snr = 10 ** ((6.02 * ENOB + 1.76) / 10.0)
        omega_BW = 2.0 * np.pi * BW
        gamma = (xi / _g_i(N) * snr) ** (1.0 / (2.0 * N))
        omega_p = omega_BW / 2.0
        beta = -omega_p * (2.0 * gamma)
        alpha = omega_p / (2.0 * gamma)
        self.T = 1.0 / abs(2.0 * omega_BW * gamma)
        kappa = beta

        A = np.zeros((N, N))
        for i in range(N):
            if i > 0:
                A[i, i - 1] = beta            # forward integration
            if i < N - 1:
                A[i, i + 1] = alpha           # leap-frog backward coupling
        self.A = A
        self.B = np.zeros((N, 1)); self.B[0, 0] = beta
        self.Gamma = kappa * np.eye(N)        # local digital control
        self.Gamma_tilde = np.eye(N)          # each state has its own comparator
        self.CT = np.eye(N)
        self.beta, self.alpha, self.kappa = beta, alpha, kappa

    # ------------------------------------------------------------------ design
    @property
    def OSR(self) -> float:
        return 1.0 / (2.0 * self.BW * self.T)

    def default_eta2(self) -> float:
        """``|G(j*omega_BW)|^2`` - the natural estimator-bandwidth parameter."""
        omega = 2.0 * np.pi * self.BW
        G = self.CT @ np.linalg.solve(1j * omega * np.eye(self.N) - self.A, self.B)
        return float(np.linalg.norm(G) ** 2)

    # ------------------------------------------------------------- modulator
    def simulate(self, u_fn: Callable[[float], float], n_samples: int,
                 x0: np.ndarray | None = None) -> np.ndarray:
        """Run the continuous-time modulator, returning control bits ``s`` in
        ``{-1, +1}`` with shape ``(n_samples, N)``.

        Each control period the state is advanced exactly (matrix exponential)
        with the input and the held control vector; the local comparators then
        observe the new state. The integral ``int_0^T expm(A sigma) dsigma`` is
        formed with the augmented-matrix trick because the leap-frog ``A`` is
        singular for odd ``N``.
        """
        A, T, N = self.A, self.T, self.N
        Ad = scipy.linalg.expm(A * T)
        aug = np.zeros((2 * N, 2 * N))
        aug[:N, :N] = A
        aug[:N, N:] = np.eye(N)
        Md = scipy.linalg.expm(aug * T)[:N, N:]          # int_0^T expm(A sigma) dsigma
        x = np.zeros(N) if x0 is None else np.array(x0, dtype=float)
        s = np.empty((n_samples, N))
        Bvec = self.B.flatten()
        for k in range(n_samples):
            sk = np.where(x >= 0.0, 1.0, -1.0)
            s[k] = sk
            drive = Bvec * u_fn(k * T) + self.Gamma @ sk
            x = Ad @ x + Md @ drive
        return s

    # ------------------------------------------------------------- estimator
    def estimator_taps(self, eta2: float | None = None, K1: int = 512, K2: int = 512) -> np.ndarray:
        """FIR estimator taps, shape ``(K1 + K2, N)``.

        ``K1`` backward-in-time (causal) taps precede ``K2`` forward (anticausal)
        taps; index ``K1`` is "now".
        """
        if eta2 is None:
            eta2 = self.default_eta2()
        A, B, Gamma, CT, T, N = self.A, self.B, self.Gamma, self.CT, self.T, self.N
        R = eta2 * np.eye(N)
        Q = B @ B.T
        Vf = scipy.linalg.solve_continuous_are(A.T, CT.T, Q, R, balanced=True)
        Vb = scipy.linalg.solve_continuous_are(-A.T, CT.T, Q, R, balanced=True)
        CCT = CT.T @ np.linalg.inv(R) @ CT
        tAf = A - Vf @ CCT
        tAb = A + Vb @ CCT
        Af = scipy.linalg.expm(tAf * T)
        Ab = scipy.linalg.expm(-tAb * T)
        W = np.linalg.lstsq(Vf + Vb, B, rcond=-1)[0]
        WT = W.T
        Bf = scipy.linalg.solve(tAf, (Af - np.eye(N)) @ Gamma)
        Bb = -scipy.linalg.solve(tAb, (Ab - np.eye(N)) @ Gamma)
        h = np.zeros((K1 + K2, N))
        Mp = np.eye(N)
        for j in range(K1):                              # forward (past) block
            h[K1 - 1 - j] = (-WT @ Mp @ Bf).flatten()
            Mp = Mp @ Af
        Mp = np.eye(N)
        for k in range(K2):                              # backward (future) block
            h[K1 + k] = (-WT @ Mp @ Bb).flatten()
            Mp = Mp @ Ab
        return h

    @staticmethod
    def reconstruct(control: np.ndarray, taps: np.ndarray, K1: int, K2: int) -> np.ndarray:
        """Estimate ``u_hat`` by correlating the control bits with the taps.

        The first ``K1`` and last ``K2`` samples are edge regions and returned as
        ``nan`` (the FIR window runs off the data there).
        """
        from numpy.lib.stride_tricks import sliding_window_view

        n = control.shape[0]
        KT = taps.shape[0]
        out = np.full(n, np.nan)
        if n < KT:
            return out
        win = sliding_window_view(control, KT, axis=0)   # (n-KT+1, N, KT): [p, m, l]
        # out[p + K1] = sum_{m,l} control[p+l, m] * taps[l, m]
        vals = np.einsum("pml,lm->p", win, taps)
        out[K1:K1 + vals.shape[0]] = vals
        return out
