# python
import json

def load_from_file(path):
    if not type(path) is str:
        path = str(path)

    data = None
    with open(path, 'r') as fh:
        data = json.load(fh)

    return data


def write_to_file(data, path, indent=None, encoding='utf-8'):
    if not type(path) is str:
        path = str(path)

    with open(path, 'w', encoding=encoding) as fh:
        json.dump(data, fh, indent=indent)