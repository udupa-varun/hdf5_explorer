from pathlib import Path

import h5py
import numpy as np

CHUNK_SIZE = int(1e4)

FILE_EXTS = ["hdf5", "h5"]


def search_for_datafiles(dir_path: Path) -> list[Path]:
    """searches given path for datafiles with valid file extensions.
    Does not search recursively.

    :param dir_path: input directory path
    :type dir_path: pathlib.Path
    :return file_paths: list of file paths
    :rtype: list[Path]
    """
    file_paths = []

    for file_ext in FILE_EXTS:
        file_paths.extend([p.resolve() for p in dir_path.glob(f"./*.{file_ext}")])

    return file_paths


def read_file(file_path: Path) -> h5py.File:
    """opens an HDF5 file in read-only mode and returns the file object.

    :param file_path: path to HDF5 file
    :type file_path: Path
    :return: read-only h5py file object for the given file path
    :rtype: h5py.File
    """
    file_obj = h5py.File(name=file_path, mode="r")

    return file_obj


def get_tasks(file_obj: h5py.File) -> list[str]:
    """fetches the DAQ task names (top level group names) from an h5py file object.

    :param file_obj: opened h5py file object
    :type file_obj: h5py.File
    :return: list of task names in file object
    :rtype: list[str]
    """
    group_names = [
        key for key in file_obj.keys() if isinstance(file_obj[key], h5py.Group)
    ]
    return group_names


def get_closest_index_before_value(
    dset: h5py.Dataset, search_val: float, chunk_size: int = CHUNK_SIZE
) -> int | None:
    """finds the index with the closest value before
    the provided search value in an h5py Dataset.
    Assumes dataset is sorted ascending.

    :param dset: h5py dataset to search inside
    :type dset: h5py.Dataset
    :param search_val: value to search for in the dataset
    :type search_val: float
    :param chunk_size: chunk size that the dataset will be broken into, defaults to 10000.
    :type chunk_size: int, optional
    :return: matching index value if found, otherwise None
    :rtype: int | None
    """
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
    """finds the index with the closest value after
    the provided search value in an h5py Dataset.
    Assumes dataset is sorted ascending.

    :param dset: h5py dataset to search inside
    :type dset: h5py.Dataset
    :param search_val: value to search for in the dataset
    :type search_val: float
    :return: matching index value if found, otherwise None
    :rtype: int | None
    """
    res = None
    idx_found = get_closest_index_before_value(dset, val)
    if idx_found is not None:
        if idx_found < dset.shape[0]:
            res = idx_found + 1
        else:
            res = idx_found

    return res


def get_group_members(group: h5py.Group) -> list[str]:
    """gets the list of children under the given h5py group.

    :param group: group to search within.
    :type group: h5py.Group
    :return: list of discovered children. These could be groups or datasets.
    :rtype: list[str]
    """
    return list(group.keys())


def get_obj_attribute(obj: h5py.Dataset | h5py.Group, attr_name: str):
    """gets the specified attribute's value for the given h5py object.

    :param obj: h5py object to check.
    :type obj: h5py.Dataset | h5py.Group
    :param attr_name: attribute label
    :type attr_name: str
    :return: attribute value(s).
    :rtype: Any | None
    """
    return obj.attrs.get(attr_name, None)
