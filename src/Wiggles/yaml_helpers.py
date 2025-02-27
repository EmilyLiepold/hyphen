import logging

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
