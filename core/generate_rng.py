"""Per-trial random streams, so every method in trial k faces the same closed-loop noise.

Trial k's streams depend only on (base_seed, k), so separate notebooks (DRMPC, DeePC)
regenerate identical noise without sharing state. Data and noise are independent
streams: changing how much data a method collects never shifts the noise.
"""

import numpy as np


def trial_rngs(base_seed, trial):
    """Return (data_rng, noise_rng) for one trial.

    Child order is fixed, so appending a stream later (spawn(3)) leaves these two unchanged.
    """
    data, noise = np.random.SeedSequence([base_seed, trial]).spawn(2) # <-- Each SeedSequence is deterministic, it will always generate the 
    return np.random.default_rng(data), np.random.default_rng(noise)
