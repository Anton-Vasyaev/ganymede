# python
from typing import List


def read_lines(
    file_path          : str,
    drop_new_line_code : bool = True
):
    with open(file_path, 'r') as fh:
        lines = fh.readlines()

    if drop_new_line_code:
        for idx in range(len(lines)):
            line = lines[idx]
            if line[-1] == '\n':
                line = line[:-1]

            lines[idx] = line

    return lines


def write_str(
    file_path : str,
    data      : str
):
    with open(file_path, 'w') as fh:
        fh.write(data)


def write_lines(
    file_path : str,
    lines     : List[str]
):
    with open(file_path, 'w') as fh:
        for line in lines:
            fh.write(f'{line}\n')


def write_bytes(
    file_path : str,
    data      : bytes
):
    with open(file_path, 'wb') as fh:
        fh.write(data)