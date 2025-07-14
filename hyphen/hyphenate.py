from .dewiggler_stack import single_dewiggle_step as single_dewiggle_step
from .dewiggler_stack import batch_dewiggle_step as batch_dewiggle_step
from .frame_merger import step as frame_merger_step
from .yaml_helpers import ingest_yaml

import argparse

parser = argparse.ArgumentParser(prog="hyphenate")
parser.add_argument("yaml_file", help="yaml_file")


def hyphenate():
    print("Hyphenating...")
    args = parser.parse_args()
    ingest_yaml(args.yaml_file, targets=[single_dewiggle_step,
                                         batch_dewiggle_step,
                                         frame_merger_step])
    print("Hyphenation complete.")
