import argparse
import pathlib

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "-l",
    "--logs",
    help="Folder path to save the logs",
    default="logs",
    type=pathlib.Path,
)
_args = _parser.parse_args()


def get_args():
    return _args
