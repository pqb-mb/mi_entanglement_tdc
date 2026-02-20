"""
Projected Quantum Encoder — Hamiltonian Time Evolution

Encodes molecular features into quantum states using MI-weighted Hamiltonian
time evolution, then extracts projected features via Pauli Z expectation values.

Implements the Kipu Quantum approach (arXiv:2510.13807):
    H(x) = Σᵢ xᵢσᵢᶻ + Σ_{(i,j)} c_{ij} σᵢᶻσⱼᶻ + Σ_{(i,j,k)} c_{ijk} σᵢᶻσⱼᶻσₖᶻ

Only the HamiltonianProjectedEncoder is included — the best-performing encoder
from hyperparameter grid search.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import pennylane as qml
import logging

logger = logging.getLogger(__name__)


class HamiltonianProjectedEncoder:
    """
    Hamiltonian-based quantum encoder using MI-weighted time evolution.

    Implements:
        H(x) = Σᵢ xᵢσᵢᶻ + Σ_{(i,j)} c_{ij} σᵢᶻσⱼᶻ + Σ_{(i,j,k)} c_{ijk} σᵢᶻσⱼᶻσₖᶻ

    where:
    - Feature values xᵢ become local field strengths (1-body terms)
    - MI scores weight interaction coupling constants (2-body and 3-body terms)
    - Time evolution exp(-iHt) encodes features into quantum state
    - Measurements extract projected features via expectation values
    """

    def __init__(self,
                 num_qubits: int = 20,
                 n_features: int = 88,
                 device_type: str = "lightning.gpu",
                 entanglement: str = "custom",
                 custom_entanglement: List[Tuple[int, int]] = None,
                 custom_triads: List[Tuple[int, int, int]] = None,
                 mi_pairs_with_scores: List[Tuple[Tuple[int, int], float]] = None,
                 mi_triads_with_scores: List[Tuple[Tuple[int, int, int], float]] = None,
                 evolution_time: float = 1.0,
                 n_trotter_steps: int = 1,
                 coupling_scale: str = "direct",
                 include_correlations: bool = True,
                 rotation_pattern: str = "ry",
                 n_cd_layers: int = 0,
                 cd_strength: float = 0.3) -> None:
        self.num_qubits = num_qubits
        self.n_features = n_features
        self.device_type = device_type
        self.evolution_time = evolution_time
        self.n_trotter_steps = n_trotter_steps
        self.coupling_scale = coupling_scale
        self.include_correlations = include_correlations
        self.n_cd_layers = n_cd_layers
        self.cd_strength = cd_strength

        # Process MI scores → coupling coefficients
        self.mi_pairs_weighted = self._process_mi_scores(
            mi_pairs_with_scores, custom_entanglement
        )
        self.mi_triads_weighted = self._process_mi_scores(
            mi_triads_with_scores, custom_triads, is_triad=True
        )

        # Output dimension: singles + pairs + triads
        if include_correlations:
            self.output_dim = (num_qubits +
                              len(self.mi_pairs_weighted) +
                              len(self.mi_triads_weighted))
        else:
            self.output_dim = num_qubits

        self._init_device()
        self._build_circuit()

        n_pairs = len(self.mi_pairs_weighted)
        n_triads = len(self.mi_triads_weighted)
        if n_pairs > 0 and n_triads == 0:
            encoder_type = "2Q (pair-based)"
        elif n_triads > 0 and n_pairs == 0:
            encoder_type = "3Q (triad-based)"
        elif n_pairs > 0 and n_triads > 0:
            encoder_type = "Hybrid (pairs + triads)"
        else:
            encoder_type = "Single-qubit only"

        cd_info = f", CD layers: {n_cd_layers}, CD strength: {cd_strength}" if n_cd_layers > 0 else ""
        logger.info(f"HamiltonianProjectedEncoder initialized ({encoder_type}):")
        logger.info(f"  Qubits: {num_qubits}, Features: {n_features}")
        logger.info(f"  Evolution time: {evolution_time}, Trotter steps: {n_trotter_steps}{cd_info}")
        logger.info(f"  Coupling scale: {coupling_scale}")
        logger.info(f"  2-body terms: {n_pairs}, 3-body terms: {n_triads}")
        logger.info(f"  Output dimension: {self.output_dim}")
        logger.info(f"  Device: {self.device_type}")

    def _process_mi_scores(self, scored_items: Optional[List[Tuple]] = None, fallback_items: Optional[List[Tuple]] = None, is_triad: bool = False) -> List[Tuple]:
        """Convert MI scores to coupling coefficients based on coupling_scale."""
        if scored_items is not None:
            items_with_mi = scored_items
        elif fallback_items is not None:
            items_with_mi = [(item, 0.1) for item in fallback_items]
        else:
            items_with_mi = []

        if not items_with_mi:
            return []

        items, scores = zip(*items_with_mi)
        scores = np.array(scores)

        if self.coupling_scale == "normalized":
            max_score = scores.max()
            if max_score > 0:
                scores = scores / max_score
        elif self.coupling_scale == "sqrt":
            scores = np.sqrt(np.abs(scores))
        elif self.coupling_scale == "weak":
            scores = scores * 0.1
        # else: "direct" - use MI scores as-is

        return list(zip(items, scores))

    def _init_device(self, max_retries: int = 3, retry_delay: int = 2) -> None:
        """Initialize PennyLane device, retrying on failure.

        Raises:
            RuntimeError: If no CUDA devices found or device creation fails.
        """
        if "gpu" in self.device_type:
            logger.info(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
            try:
                import ctypes
                libcuda = ctypes.cdll.LoadLibrary("libcuda.so.1")
                rc_init = libcuda.cuInit(0)
                count = ctypes.c_int(0)
                rc_count = libcuda.cuDeviceGetCount(ctypes.byref(count))
                logger.info(f"  CUDA driver: cuInit={rc_init}, cuDeviceGetCount={count.value} (rc={rc_count})")
                if count.value == 0:
                    raise RuntimeError(
                        "cuDeviceGetCount=0 -- no CUDA devices visible. "
                        "Check CUDA_VISIBLE_DEVICES and nvidia-smi."
                    )
                # Verify CUDA runtime API (not just driver) sees the GPU.
                # If PyTorch stubs poisoned the runtime, this will catch it
                # before PennyLane Lightning's C++ code hits DevTag::refresh().
                try:
                    cudart = None
                    for rt_name in ("libcudart.so.12", "libcudart.so.11"):
                        try:
                            cudart = ctypes.cdll.LoadLibrary(rt_name)
                            break
                        except OSError:
                            continue
                    if cudart is not None:
                        rt_count = ctypes.c_int(0)
                        rc_rt = cudart.cudaGetDeviceCount(ctypes.byref(rt_count))
                        rc_set = cudart.cudaSetDevice(0)
                        logger.info(f"  CUDA runtime: cudaGetDeviceCount={rt_count.value} (rc={rc_rt}), cudaSetDevice(0) rc={rc_set}")
                    else:
                        logger.warning("  CUDA runtime: libcudart.so.12 not found")
                except Exception as e:
                    logger.warning(f"  CUDA runtime check failed: {e}")
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"CUDA driver check failed: {e}") from e

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                self.dev = qml.device(self.device_type, wires=self.num_qubits)
                logger.info(f"  SUCCESS: Created {self.device_type} device ({self.num_qubits} wires)")
                return
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    logger.warning(f"  Attempt {attempt}/{max_retries} failed: {e}")
                    logger.warning(f"  Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)

        raise RuntimeError(
            f"Failed to create {self.device_type} after {max_retries} attempts: {last_err}"
        )

    def _build_circuit(self) -> None:
        """Build Hamiltonian evolution circuit, optionally with counterdiabatic driving.

        When n_cd_layers == 0: standard Hamiltonian evolution (backward compatible).
        When n_cd_layers > 0: interleave Hamiltonian evolution with data-dependent
        Y rotations (counterdiabatic correction from adiabatic gauge potential).

        The CD correction breaks the diagonal structure of the all-Z Hamiltonian,
        enabling the circuit to explore more of Hilbert space.
        """
        num_qubits = self.num_qubits
        n_features = self.n_features
        evolution_time = self.evolution_time
        n_trotter_steps = self.n_trotter_steps
        mi_pairs = self.mi_pairs_weighted
        mi_triads = self.mi_triads_weighted
        include_correlations = self.include_correlations
        n_cd_layers = self.n_cd_layers
        cd_strength = self.cd_strength

        @qml.qnode(self.dev)
        def circuit(x):
            coeffs = []
            obs = []

            # 1-body terms: H₁ = Σᵢ xᵢ σᵢᶻ
            for i in range(num_qubits):
                if i < n_features:
                    coeffs.append(x[i])
                    obs.append(qml.PauliZ(i))

            # 2-body terms: H₂ = Σ_{(i,j)} c_{ij} σᵢᶻ σⱼᶻ
            for (i, j), coupling in mi_pairs:
                coeffs.append(coupling)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

            # 3-body terms: H₃ = Σ_{(i,j,k)} c_{ijk} σᵢᶻ σⱼᶻ σₖᶻ
            for (i, j, k), coupling in mi_triads:
                coeffs.append(coupling)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j) @ qml.PauliZ(k))

            H = qml.Hamiltonian(coeffs, obs)

            # Initialize qubits in superposition
            for i in range(num_qubits):
                qml.Hadamard(wires=i)

            if n_cd_layers > 0:
                # Counterdiabatic driving: interleave Hamiltonian evolution
                # with data-dependent Y rotations (adiabatic gauge potential)
                dt = evolution_time / n_cd_layers
                for layer in range(n_cd_layers):
                    # Hamiltonian evolution for time slice dt
                    qml.ApproxTimeEvolution(H, dt, n_trotter_steps)
                    # CD correction: RY(α · xᵢ) on each qubit
                    for i in range(num_qubits):
                        if i < n_features:
                            qml.RY(cd_strength * x[i], wires=i)
            else:
                # Standard: single Hamiltonian evolution (backward compatible)
                qml.ApproxTimeEvolution(H, evolution_time, n_trotter_steps)

            # Measurements
            measurements = []

            # Single-qubit Z expectations
            for i in range(num_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))

            if include_correlations:
                for (i, j), _ in mi_pairs:
                    measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
                for (i, j, k), _ in mi_triads:
                    measurements.append(
                        qml.expval(qml.PauliZ(i) @ qml.PauliZ(j) @ qml.PauliZ(k))
                    )

            return measurements

        self.circuit = circuit

    def encode(self, x: np.ndarray, _max_retries: int = 3) -> np.ndarray:
        """Encode single feature vector → projected features."""
        x = np.asarray(x, dtype=np.float64)
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")

        for attempt in range(_max_retries):
            try:
                expectations = self.circuit(x)
                return np.array(expectations, dtype=np.float32)
            except Exception as e:
                if "busy or unavailable" in str(e) and attempt < _max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(f"[GPU busy, retry {attempt+1}/{_max_retries} in {wait}s] {type(e).__name__}: {e}")
                    time.sleep(wait)
                else:
                    raise

    def encode_batch(self,
                     features: np.ndarray,
                     show_progress: bool = True,
                     n_workers: int = 1) -> np.ndarray:
        """Encode batch of feature vectors."""
        n_samples = features.shape[0]

        if n_workers == 1:
            output = np.zeros((n_samples, self.output_dim), dtype=np.float32)
            iterator = range(n_samples)
            if show_progress:
                iterator = tqdm(iterator, desc=f"Hamiltonian Encoding ({self.num_qubits}q)")

            for i in iterator:
                output[i] = self.encode(features[i])

            return output

        else:
            import multiprocessing as mp
            ctx = mp.get_context('spawn')

            _src_dir = os.path.dirname(os.path.abspath(__file__))
            if _src_dir not in sys.path:
                sys.path.insert(0, _src_dir)
            from encoder_worker import encode_chunk_worker

            encoder_config = {
                'num_qubits': self.num_qubits,
                'n_features': self.n_features,
                'device_type': self.device_type,
                'entanglement': 'custom',
                'custom_entanglement': [pair for pair, _ in self.mi_pairs_weighted],
                'custom_triads': [triad for triad, _ in self.mi_triads_weighted],
                'mi_pairs_with_scores': self.mi_pairs_weighted,
                'mi_triads_with_scores': self.mi_triads_weighted,
                'evolution_time': self.evolution_time,
                'n_trotter_steps': self.n_trotter_steps,
                'coupling_scale': self.coupling_scale,
                'include_correlations': self.include_correlations,
                'n_cd_layers': self.n_cd_layers,
                'cd_strength': self.cd_strength,
            }

            chunks = np.array_split(features, n_workers)
            chunk_args = [(chunk, encoder_config, "HamiltonianProjectedEncoder")
                         for chunk in chunks]

            if show_progress:
                logger.info(f"Hamiltonian Encoding {n_samples} samples with {n_workers} workers (spawn)...")

            with ctx.Pool(n_workers) as pool:
                results = pool.map(encode_chunk_worker, chunk_args)

            return np.vstack(results)
