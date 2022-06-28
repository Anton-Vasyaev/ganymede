def is_any(value, compare_values):
    for compare_value in compare_values:
        if value == compare_value: 
            return True

    return False