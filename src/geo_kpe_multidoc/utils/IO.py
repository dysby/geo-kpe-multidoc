import pickle


def write_to_file(filename: str = "", con: dict = {}):
    """
    Small auxiliary function to write data to a file
    """
    with open("{}.txt".format(filename), "wb") as write_f:
        pickle.dump(con, write_f)
    return


def read_from_file(filename: str = ""):
    """
    Small auxiliary function to read data from a file
    """
    with open("{}.txt".format(filename), "rb") as read_f:
        return pickle.load(read_f)
