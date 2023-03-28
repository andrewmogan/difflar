import os
import argparse

def check_input_filename(filename):
    """
    Checks if the given input file has a .root extension.
    Raises an argparse.ArgumentTypeError if the extension is invalid.
    """

    extension = '.root'
    _, input_extension = os.path.splitext(filename)
    if input_extension != extension:
        raise argparse.ArgumentTypeError("Input must be a .root file")
    return filename

def check_input_range(range):

    if range[0] > range[1]:
        raise argparse.ArgumentTypeError("Input range min must be less than max in {}".format(range))
    if range[2] <= 0:
        raise argparse.ArgumentTypeError("Input step size must be positive float (greater than 0) in {}".format(range))
