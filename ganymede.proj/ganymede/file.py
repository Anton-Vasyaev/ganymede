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


def write_bytes(
    file_path : str,
    data : bytes
):
    with open(file_path, 'wb') as fh:
        fh.write(data)