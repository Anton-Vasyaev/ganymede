def is_none_or_empty(str_data : str, argument_name : str):
    valid = True
    if str_data is None: raise Exception(
        f'str argument \'{argument_name}\' was None'
    )

    if str_data == '':  raise Exception(
        f'str agument \'{argument_name}\' was empty'
    )
    