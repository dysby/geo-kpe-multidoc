import os
import pickle
import re
from pathlib import Path


def write_to_file(filename: str = "", content: dict = {}):
    """
    Small auxiliary function to write data to a file
    """
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    # pickle.dump(content, file=f"{filename}.txt")
    with open("{}.txt".format(filename), "wb") as write_f:
        pickle.dump(content, write_f)


def read_from_file(filename: str = ""):
    """
    Small auxiliary function to read data from a file
    """
    content = None
    with open("{}.txt".format(filename), "rb") as read_f:
        content = pickle.load(read_f)

    return content


def get_mdkpe_files(path: str):
    """Generate a list of files in dir that follow the pattern `dnn-mdkpe-geo.pkl`.
    The files where saved from MDKPERank model outputs.

    Parameters
    ----------
    path : str
        Directory to search

    Yields
    ------
    str
        file name
    """
    geo_file_name_pattern = re.compile(r"d\d{2}-mdkpe-geo\.pkl")
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and geo_file_name_pattern.match(
            file
        ):
            yield file
