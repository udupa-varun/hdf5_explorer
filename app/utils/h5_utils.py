from pathlib import Path

import h5py
import numpy as np

CHUNK_SIZE = int(1e4)

FILE_EXTS = [".hdf5", ".h5"]


def search_for_datafiles(dir_path: Path) -> list[Path]:
    """searches given path for datafiles of valid types

    :param dir_path: input directory path
    :type dir_path: pathlib.Path
    :return file_paths: list of file paths
    :rtype: list[Path]
    """
    file_paths = []

    file_paths = list(
        p.resolve() for p in dir_path.glob("**/*") if p.suffix in FILE_EXTS
    )
    return file_paths


def read_file(file_path: Path) -> h5py.File:
    file_obj = h5py.File(name=file_path, mode="r")

    return file_obj


def get_tasks(file_obj: h5py.File) -> list[str]:
    # global file_obj
    return list(file_obj.keys())


def get_closest_index_before_value(
    dset: h5py.Dataset, search_val, chunk_size=CHUNK_SIZE
) -> int | None:
    num_chunks = int(np.ceil(dset.shape[0] / chunk_size))
    idx_found = None
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_stop = min(chunk_start + chunk_size, dset.shape[0])

        matches = np.argwhere(dset[chunk_start:chunk_stop] >= search_val)

        # exit condition
        if matches.any():
            idx_found = matches[0][0]
            break

    return idx_found


def get_closest_index_after_value(dset: h5py.Dataset, val) -> int | None:
    res = None
    idx_found = get_closest_index_before_value(dset, val)
    if idx_found is not None:
        if idx_found < dset.shape[0]:
            res = idx_found + 1
        else:
            res = idx_found

    return res
