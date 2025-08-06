#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code was initially inspired by the following :
https://github.com/machow/pysearchlight

@author: Daniel Lindh
"""
from collections.abc import Hashable, Sequence
from copy import deepcopy
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.data.noise import prec_from_unbalanced
from rsatoolbox.rdm import RDMs
from rsatoolbox.rdm.calc import calc_rdm
from scipy.spatial.distance import cdist
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def _get_searchlight_neighbors(mask, center, radius=3):
    """Return indices for searchlight where distance
        between a voxel and their center < radius (in voxels)

    Args:
        center (index):  point around which to make searchlight sphere

    Returns:
        list: the list of volume indices that respect the
                searchlight radius for the input center.
    """
    center = np.array(center)
    mask_shape = mask.shape
    cx, cy, cz = np.array(center)
    x = np.arange(mask_shape[0])
    y = np.arange(mask_shape[1])
    z = np.arange(mask_shape[2])

    # First mask the obvious points
    # - may actually slow down your calculation depending.
    x = x[abs(x - cx) < radius]
    y = y[abs(y - cy) < radius]
    z = z[abs(z - cz) < radius]

    # Generate grid of points
    X, Y, Z = np.meshgrid(x, y, z)
    data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    distance = cdist(data, center.reshape(1, -1), "euclidean").ravel()

    return tuple(data[distance < radius].T.tolist())


def get_volume_searchlight(mask, radius=2, threshold=1.0):
    """
    Searches through the non-zero voxels of the mask, selects centers where
    proportion of sphere voxels >= self.threshold.

    Args:

        mask ([numpy array]): binary brain mask

        radius (int, optional): the radius of each searchlight, defined in voxels.
        Defaults to 2.

        threshold (float, optional): Threshold of the proportion of voxels that need to
        be inside the brain mask in order for it to be
        considered a good searchlight center.
        Values go between 0.0 - 1.0 where 1.0 means that
        100% of the voxels need to be inside
        the brain mask.
        Defaults to 1.0.

    Returns:
        numpy array: array of centers of size n_centers x 3

        list: list of lists with neighbors - the length of the list will correspond to:
        n_centers x 3 x n_neighbors
    """

    mask = np.array(mask)
    assert mask.ndim == 3, "Mask needs to be a 3-dimensional numpy array"

    centers = list(zip(*np.nonzero(mask)))
    good_centers = []
    good_neighbors = []

    for center in tqdm(centers, desc="Finding searchlights..."):
        neighbors = _get_searchlight_neighbors(mask, center, radius)
        if mask[neighbors].mean() >= threshold:
            good_centers.append(center)
            good_neighbors.append(neighbors)

    good_centers = np.array(good_centers)
    assert good_centers.shape[0] == len(
        good_neighbors
    ), "number of centers and sets of neighbors do not match"
    print(f"Found {len(good_neighbors)} searchlights")

    # turn the 3-dim coordinates to array coordinates
    centers = np.ravel_multi_index(good_centers.T, mask.shape)
    neighbors = [np.ravel_multi_index(n, mask.shape) for n in good_neighbors]

    return centers, neighbors


def _get_chunk_searchlight_RDMs(
    chunks,
    data_2d,
    centers,
    neighbors,
    events: Sequence[Hashable],
    method: str,
    cv_descriptor: Sequence[Hashable] = None,
    noise: str | Sequence[np.ndarray] | None = None,
    **calcKwargs,
) -> RDMs:
    """Calculates the RDMs for a chunk of searchlight centers.

    Args:
        chunks (list): List of chunked center indices.

        data_2d (2D numpy array): Flattened brain data, n_observations x n_channels.

        noise (str | Sequence[np.ndarray], optional): Noise estimation method name to estimate the
        noise with rsatoolbox.data.noise.prec_from_unbalanced OR a list of precision matrixes with
        one for each center in chunks.

        See get_searchlight_RDMs for the other arguments.

    Returns:
        RDM [rsatoolbox.rdm.RDMs]: RDMs for the chunk of centers.
    """
    calcKwargs = deepcopy(calcKwargs)
    obs_descriptors = {"events": events}
    if cv_descriptor:
        # Add cv descriptor to Dataset
        obs_descriptors |= {"cv_descriptor": cv_descriptor}
        # Ensure it is used by calc_rdm
        calcKwargs["cv_descriptor"] = "cv_descriptor"
    center_data = []
    for c in chunks:
        # grab this center and neighbors
        center = centers[c]
        center_neighbors = neighbors[c]
        # create a database object with this data
        ds = Dataset(
            data_2d[:, center_neighbors],
            descriptors={"center": center},
            obs_descriptors=obs_descriptors,
            channel_descriptors={"voxels": center_neighbors},
        )
        center_data.append(ds)
    if noise:
        if isinstance(noise, str):
            # If noise is a string, use the method to estimate noise
            # This calculates the precision matrix for each center_data dataset
            calcKwargs["noise"] = prec_from_unbalanced(
                center_data, obs_desc="events", method=noise
            )
        elif isinstance(noise, Sequence):
            if len(noise) != len(center_data):
                raise ValueError("Noise sequence length must match chunks length.")
            else:
                calcKwargs["noise"] = noise

    return calc_rdm(center_data, method=method, descriptor="events", **calcKwargs)


def get_searchlight_RDMs(
    data,
    centers,
    neighbors,
    events,
    method="correlation",
    cv_descriptor: Sequence[Hashable] = None,
    noise_method: str = None,
    batchsize=100,
    maxWorkers=1,
    **calc_rdm_kwargs,
) -> RDMs:
    """Iterates over all the searchlight centers and calculates the RDM

    Args:

        data (2D or 4D numpy array): brain data,
        either flattened shape n_observations x n_channels (i.e. voxels/vertices),
        or 4D shape n_observations x x_dim x y_dim x z_dim. Should match the mask used to create
        centers and neighbors.

        centers (1D numpy array): center indices for all searchlights as provided
        by rsatoolbox.util.searchlight.get_volume_searchlight

        neighbors (list): list of lists with neighbor voxel indices for all searchlights
        as provided by rsatoolbox.util.searchlight.get_volume_searchlight

        events (1D numpy array): 1D array of length n_observations

        method (str, optional): distance metric,
        see rsatoolbox.rdm.calc for options. Defaults to 'correlation'.

        cv_descriptor (1D array-like, optional): if provided, n_observations length sequence of
        cross-validation coding for each observation. Defaults to no cross-validation.

        noise_method (str, optional): if provided, the method to use for estimating noise with
        rsatoolbox.data.noise.prec_from_unbalanced at each center. Only valid if method is
        'mahalanobis' or 'crossnobis'.

        batchsize (int, optional): Searchlight center processing batch size. Defaults to 100.

        maxWorkers (int, optional): maximum number of parallel workers. Defaults to 1, i.e. no
        parallel processing.

        calc_rdm_kwargs (dict, optional): additional keyword arguments to pass to
        rsatoolbox.rdm.calc.calc_rdm.

    Returns:
        RDM [rsatoolbox.rdm.RDMs]: RDMs object with the RDM for each searchlight
                              the RDM.rdm_descriptors['voxel_index']
                              describes the center voxel index each RDM is associated with
    """
    if data.ndim == 4:
        data = data.reshape((len(events), -1))
    data_2d, centers = np.array(data), np.array(centers)
    n_centers = centers.shape[0]

    # we can't run all centers at once, that will take too much memory
    # so lets to some chunking
    chunked_center = np.array_split(np.arange(n_centers), n_centers // batchsize + 1)

    # loop over chunks
    n_conds = len(np.unique(events))
    fixed_parallel_function = partial(
        _get_chunk_searchlight_RDMs,
        data_2d=data_2d,
        centers=centers,
        neighbors=neighbors,
        events=events,
        method=method,
        cv_descriptor=cv_descriptor,
        noise=noise_method,
        **calc_rdm_kwargs,
    )
    if len(chunked_center) == 1 or maxWorkers == 1:
        # if we have only one chunk or no parallel processing, run it directly
        RDM_corrs = [
            fixed_parallel_function(chunks)
            for chunks in tqdm(chunked_center, desc="Calculating RDMs...")
        ]
    else:
        # TODO: Tune this chunking. Even with the default of 100 centers per chunked_center, the
        # overhead per process is too high to give only one to each TQDM process. Modified from
        # https://stackoverflow.com/a/42096963/5722359, which is for a different task.
        parallelChunkSize = 20 + (
            int(len(chunked_center) / maxWorkers / 5) if len(chunked_center) > 1000 else 0
        )
        RDM_corrs = process_map(
            fixed_parallel_function,
            chunked_center,
            desc="Calculating RDMs...",
            max_workers=maxWorkers,
            chunksize=parallelChunkSize,
            smoothing=0,
        )

    # Collect the results from all chunks
    RDM = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
    rdm_events: list[str] = []
    for chunks, RDM_corr in zip(chunked_center, RDM_corrs):
        if rdm_events:
            assert (
                rdm_events == RDM_corr.pattern_descriptors["events"]
            ), "RDMs from different chunks have different event descriptors."
        else:
            rdm_events = RDM_corr.pattern_descriptors["events"]
        RDM[chunks, :] = RDM_corr.dissimilarities

    SL_rdms = RDMs(
        RDM,
        rdm_descriptors={"voxel_index": centers},
        dissimilarity_measure=method,
        pattern_descriptors={"events": rdm_events},
    )

    return SL_rdms


def evaluate_models_searchlight(
    sl_RDM, models, eval_function, method="corr", theta=None, n_jobs=1
):
    """evaluates each searchlighth with the given model/models

    Args:

        sl_RDM ([rsatoolbox.rdm.RDMs]): RDMs object
        as computed by rsatoolbox.util.searchlight.get_searchlight_RDMs

        models ([rsatoolbox.model]: models to evaluate - can also be list of models

        eval_function (rsatoolbox.inference evaluation-function): [description]

        method (str, optional): see rsatoolbox.rdm.compare for specifics. Defaults to 'corr'.

        n_jobs (int, optional): how many jobs to run. Defaults to 1.

    Returns:

        list: list of with the model evaluation for each searchlight center
    """

    results = Parallel(n_jobs=n_jobs)(
        delayed(eval_function)(models, x, method=method, theta=theta)
        for x in tqdm(sl_RDM, desc="Evaluating models for each searchlight")
    )

    return results
