#!/usr/bin/env python

import h5py 
import bioblend
#import tensorflow as tf
import argparse
import time

def print_script(args):
    print("Found script")
    print(bioblend.__version__)
    print(h5py.__version__)
    import tensorflow as tf
    print(tf.__version__)
    print(args)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-ldf", "--loaded_file", required=True, help="")
    arg_parser.add_argument("-om", "--output_model", required=True, help="")
    arg_parser.add_argument("-oa", "--output_array", required=True, help="")

    # get argument values
    args = vars(arg_parser.parse_args())
    print_script(args)
