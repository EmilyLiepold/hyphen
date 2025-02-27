__version__ = "0.0.1"

from .dewiggler_stack import step as dewiggler_step
from .frame_merger import step as frame_merger_step
from .yaml_helpers import ingest_yaml


__all__ = ["dewiggler_wrapper"]

def hyphenate(args):
    print("Hyphenating...")
    print(args)

    ingest_yaml(args, targets=[dewiggler_step, frame_merger_step])
    print("Hyphenation complete.")
