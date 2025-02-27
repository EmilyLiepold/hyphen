from tqdm import tqdm


def wigglebar(*args, **kwargs):
    if not args:
        return _tqdm(**kwargs, bar_format="|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]", ascii='~|-')
    return _tqdm(*args, **kwargs, bar_format="|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]", ascii='~|-')