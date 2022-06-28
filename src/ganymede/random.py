import random

def provide_default_instance_if_none(random_instance):
    if random_instance is None: return random
    else: return random_instance


def get_random_distance(
    start = 0.0, 
    end = 1.0,
    random_instance = None
):
    random_instance = provide_default_instance_if_none(random_instance)

    distance = end - start

    rand_val = start + (random_instance.random() * distance)

    return rand_val




def get_random_bool(
    true_border = 0.5,
    random_instance = None
):
    random_instance = provide_default_instance_if_none(random_instance)

    val = random_instance.random()
    if val > true_border: return True
    else: return False


def get_random_enum(
    enum_type,
    random_instance = None
):
    random_instance = provide_default_instance_if_none(random_instance)

    return enum_type(int(get_random_distance(0, len(enum_type))))


def choice(
    data,
    random_instance = None
):
    random_instance = provide_default_instance_if_none(random_instance)

    return random_instance.choice(data)


def sample(
    data,
    len,
    random_instance = None
):
    random_instance = provide_default_instance_if_none(random_instance)

    return random_instance.sample(data, len)