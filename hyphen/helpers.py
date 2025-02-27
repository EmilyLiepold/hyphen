from tqdm import tqdm


def wigglebar(*args, **kwargs):
    if not args:
        return tqdm(**kwargs, bar_format="|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]", ascii='~◊-')
    return tqdm(*args, **kwargs, bar_format="|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]", ascii='~◊-')