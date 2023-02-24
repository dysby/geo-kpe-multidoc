import pickle
from pathlib import Path


def write_to_file(filename: str = "", con: dict = {}):
    """
    Small auxiliary function to write data to a file
    """
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    pickle.dump(file=f"{filename}.txt")
    # with open("{}.txt".format(filename), "wb") as write_f:
    #    pickle.dump(con, write_f)


def read_from_file(filename: str = ""):
    """
    Small auxiliary function to read data from a file
    """
    with open("{}.txt".format(filename), "rb") as read_f:
        pickle.load(read_f)
