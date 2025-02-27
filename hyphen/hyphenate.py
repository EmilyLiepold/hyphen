from .dewiggler_stack import step as dewiggler_step
from .frame_merger import step as frame_merger_step
from .yaml_helpers import ingest_yaml

import argparse

parser = argparse.ArgumentParser(prog="hyphenate")
parser.add_argument("yaml_file", help="yaml_file")

def main():
    
    print(args.echo)


def hyphenate():
    print("Hyphenating...")
    args = parser.parse_args()
    print(args)

    # ingest_yaml(args, targets=[dewiggler_step, frame_merger_step])
    print("Hyphenation complete.")
