import os
import numpy as np


def get_filename_list(file_dir):
    """ Append file name strings ending with '.csv' into an empty list """
    csv_list = [files for files in sorted(os.listdir(file_dir))
                if files.endswith(".csv")]

    return csv_list

