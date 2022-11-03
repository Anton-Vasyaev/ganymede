def less_to_grt(a, b, eps = 1e-6):
    min_v = min(a, b)
    max_v = max(a, b)

    return min_v / (max_v + eps)


def grt_to_less(a, b, eps = 1e-6):
    min_v = min(a, b)
    max_v = max(a, b)

    return max_v / (min_v + eps)



def equal_err(a, b, eps = 1e-6):
    diff = abs(a - b)

    return less_to_grt(diff, max(a, b)) < eps