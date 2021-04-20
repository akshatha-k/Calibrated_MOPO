_TRAINERS = dict()


def register(fn):
    global _TRAINERS
    _TRAINERS[fn.__name__] = fn()
    return fn


def get_config(name=None):
    if name is None:
        return _TRAINERS
    return _TRAINERS[name]
