import logging
import yaml

logger = logging.getLogger(__name__)


def validate_yaml_arguments(yaml_files, necessary_arguments=()):
    """Check names occur in a mapping; values, including None, are not validated.

    Log the first missing name and return False; callers decide whether to skip.
    """

    for arg in necessary_arguments:
        found_arg = False
        for yaml_file in yaml_files:
            if arg in yaml_file.keys():
                found_arg = True

        if not found_arg:
            logger.error(f"Argument {arg} is missing from the yaml file.")
            return found_arg

    return True


def yaml_layerer(yaml_files):
    """Return a getter applying fallback, layers, then per-call overrides.

    Mappings have increasing precedence; a present None replaces earlier values.
    """

    def generic_get(arg, default=None, override=()):
        out = default
        for yaml_file in yaml_files:
            out = yaml_file.get(arg, out)

        for yaml_file in override:
            out = yaml_file.get(arg, out)

        return out

    return generic_get


def ingest_yaml(yaml_file, targets=[]):
    """Dispatch mapping documents from a YAML path to matching targets in order.

    A generic document replaces shared arguments for subsequent steps. Missing
    or unknown step types are ignored; missing required names log and skip the
    matching target. Required names are checked in generic/step mappings only.
    Dispatch needs a step-local label. Target errors and invalid document shapes
    are not handled here.
    """

    with open(yaml_file, "r") as file:
        docs = [dict(d) for d in yaml.safe_load_all(file)]
        generic_args = {}
        for doc in docs:
            if "step_type" not in doc.keys():
                continue
            if doc["step_type"] == "generic":
                generic_args = doc
                continue

            for t in targets:
                if doc["step_type"] == t["target_step_type"]:
                    necessary_arguments = t.get("necessary_arguments", [])
                    default_options = t.get("default_options", {})
                    target_function = t.get("target_function")

                    if not validate_yaml_arguments(
                        [generic_args, doc],
                        necessary_arguments=necessary_arguments,
                    ):
                        continue

                    print(f'Beginning step {doc["label"]}')

                    target_function(yaml_layerer([default_options, generic_args, doc]))
