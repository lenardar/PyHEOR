"""Optional process execution for independent cohort PSA draws."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

from ..distributions import sample_distribution


_worker_model = None


def validate_n_jobs(n_jobs):
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")


def _initialize_worker(model_bytes):
    import cloudpickle
    global _worker_model
    _worker_model = cloudpickle.loads(model_bytes)


def _simulate_draw(params):
    import cloudpickle
    with _worker_model._attr_param_override(params):
        result = _worker_model._simulate_single(params)
    # PSM results may retain user-defined survival distribution classes.
    return cloudpickle.dumps(result)


def run_parallel_psa(model, n_sim, seed, progress, n_jobs):
    """Sample in the parent and collect complete draws in their original order."""
    import cloudpickle
    from ..analysis.results import PSAResult

    rng = np.random.default_rng(seed)
    sampled_params = []
    for _ in range(n_sim):
        params = model._get_base_params()
        for name, parameter in model.params.items():
            if parameter.dist is not None:
                params[name] = float(sample_distribution(parameter.dist, 1, rng)[0])
        sampled_params.append(params)

    model_bytes = cloudpickle.dumps(model)
    results = []
    with ProcessPoolExecutor(
        max_workers=min(int(n_jobs), n_sim),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(model_bytes,),
    ) as pool:
        # Batch queue traffic without splitting strategies within a draw.
        for i, payload in enumerate(pool.map(_simulate_draw, sampled_params, chunksize=8)):
            results.append(cloudpickle.loads(payload))
            if progress and (i + 1) % max(1, n_sim // 10) == 0:
                print(f"  PSA: {i+1}/{n_sim} ({100*(i+1)/n_sim:.0f}%)")
    if progress:
        print(f"  PSA complete: {n_sim} simulations")
    return PSAResult(model=model, psa_results=results, sampled_params=sampled_params)
