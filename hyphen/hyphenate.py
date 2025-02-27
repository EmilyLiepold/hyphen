from .dewiggler_stack import step as dewiggler_step
from .frame_merger import step as frame_merger_step
from .yaml_helpers import ingest_yaml


def hyphenate(args):
    print("Hyphenating...")
    print(args)

    # ingest_yaml(args, targets=[dewiggler_step, frame_merger_step])
    print("Hyphenation complete.")
