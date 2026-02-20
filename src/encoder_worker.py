"""
Multiprocessing worker for quantum encoding.

IMPORTANT: This module must NOT import pennylane at the top level.
The worker function sets CUDA_VISIBLE_DEVICES before importing PennyLane
to ensure proper GPU initialization in spawned child processes.
"""
from __future__ import annotations


def encode_chunk_worker(args: tuple) -> np.ndarray:
    """Worker function for multiprocessing encode_batch.

    IMPORTANT: Set CUDA_VISIBLE_DEVICES before importing PennyLane to ensure
    GPU is properly initialized in spawned child processes.

    Args:
        args: Tuple of (chunk_features, encoder_config, encoder_class)

    Returns:
        Encoded features for this chunk
    """
    import os
    import time
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # CRITICAL: Set CUDA env vars BEFORE any PennyLane imports
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"

    import numpy as np
    import pennylane as qml

    chunk_features, encoder_config, encoder_class = args
    pid = os.getpid()

    import sys as _sys
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    if _src_dir not in _sys.path:
        _sys.path.insert(0, _src_dir)
    from projected_q_encoder import HamiltonianProjectedEncoder

    if encoder_class == "HamiltonianProjectedEncoder":
        encoder = HamiltonianProjectedEncoder(**encoder_config)
    else:
        raise ValueError(f"Unknown encoder class: {encoder_class}")

    actual_device = encoder.dev.name
    logger.info("[Worker %d] %s using device: %s", pid, encoder_class, actual_device)

    n_samples = chunk_features.shape[0]
    output = np.zeros((n_samples, encoder.output_dim), dtype=np.float32)
    start_time = time.time()
    for i in range(n_samples):
        output[i] = encoder.encode(chunk_features[i])
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            logger.info("[Worker %d] %d/%d (%.1fs)", pid, i + 1, n_samples, elapsed)

    elapsed = time.time() - start_time
    logger.info("[Worker %d] %d/%d done (%.1fs)", pid, n_samples, n_samples, elapsed)

    return output
