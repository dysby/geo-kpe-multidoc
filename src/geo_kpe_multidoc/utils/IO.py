import pickle
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
