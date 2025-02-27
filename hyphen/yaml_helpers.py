import logging
import yaml

logger = logging.getLogger(__name__)


def validate_yaml_arguments(yaml_files, necessary_arguments=()):

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

    def generic_get(arg, default=None):
        out = default
        for yaml_file in yaml_files:
            out = yaml_file.get(arg, out)

        return out

    return generic_get

def ingest_yaml(yaml_file, targets = []):

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
                if doc["step_type"] == t['target_step_type']:
                    necessary_arguments = t.get('necessary_arguments',[])
                    default_options = t.get('default_options',{})
                    target_function = t.get('target_function')

                    if not validate_yaml_arguments(
                        [generic_args, doc],
                        necessary_arguments=necessary_arguments,
                    ):
                        continue

                    print(f'Beginning step {doc["label"]}')
                    
                    target_function(yaml_layerer([default_options, generic_args, doc]))

                    

