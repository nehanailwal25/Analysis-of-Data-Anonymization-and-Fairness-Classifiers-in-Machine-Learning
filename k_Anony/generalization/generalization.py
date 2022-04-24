# -*- coding: utf-8 -*-

import math
from generalization.hierarchy_utilities import read_gen_hierarchy


def hierarchy(path: str, qi_name: str)-> list:
    return [substitution, read_gen_hierarchy(path, qi_name)]


def age(data, level):
    # Use the generic function "segmentation" with predefined arguments
    return segmentation(data, level, 1, 100, [5, 10, 20, "*"])


def segmentation(data, level, min_num, max_num, div_list):
    ret = []
    # Check if data is already a list/range or if it is a single value
    if not isinstance(data, list) and not isinstance(data, range):
        values = [int(data)]
    else:
        values = list(map(int, data))

    seg = div_list[level]

    # Check if the last level is not an integer segmentation and thus a substitution
    if len(div_list)-1 == level and not isinstance(seg, int):
        return l1sub(values, seg)

    groups = range(0, math.floor((max_num+1-min_num)/seg))
    div_max = min_num + seg + seg * groups[-1]

    for value in values:
        # Check if a value is bigger than the calculated max segementation value
        if value >= div_max:
            # Cut larger value to fit the segementation
            value = div_max - 1

        # Check in what group the value belongs
        for i in groups:
            b = min_num + seg * i
            e = b + seg
            if b <= value < e:
                e -= 1
                ret.append(str(b) + "-" + str(e))
                break
    return ret


def zip_code(data, level):
    """Transforms zipcode data to a predefined generalization state."""
    # Use the generic function "removeal" with predefined arguments
    return removeal(data, level, 1)


def removeal(data, level, steps):
    """Transforms zipcode data to a generalization state with removed characters."""
    ret = []
    # Check if data is already a list or if it is a single value
    if not isinstance(data, list):
        values = [data]
    else:
        values = data

    # How many characters to remove this level
    char_num = (level+1)*steps

    # Check if every character would be removed
    if char_num >= len(str(values[0])):
        return l1sub(values, level)

    for v in values:
        v = list(str(v))
        # Replace every character that gets removed with *
        for n in range(char_num):
            v[(-1-n)] = '*'
        ret.append("".join(v))
    return ret


def birthdate(data, level, min_year, max_year):
    """Transforms birthdate data to a predefined generalization state."""
    ret = []
    if not isinstance(data, list):
        values = [data]
    else:
        values = data

    # Remove parts of date string
    for v in values:
        ret.append(v.split(".", level + 1)[-1])

    # If last generalization level is reached, apply segementation of the year
    if level >= 2:
        ret = list(map(int, ret))
        ret = segmentation(ret, 0, min_year, max_year, [10])

    return ret


def l1sub(data, placeholder):
    """Substitutes data with a character (default: *)."""
    if isinstance(placeholder, int):
        sub_char = '*'
    else:
        sub_char = placeholder

    if not isinstance(data, list):
        values = [data]
    else:
        values = data

    return [sub_char]*len(values)


def substitution(data, level, wordlists):
    """Transforms birthdate data to a generalization state with substituted values."""
    ret = []
    if not isinstance(data, list):
        values = [data]
    else:
        values = data

    # Check if no more substitution is found
    if level > len(wordlists)-1:
        return l1sub(data, level)

    # Select right dictionary
    wordlist = wordlists[level]

    for value in values:
        # Search for value in dictionary
        for k, v in wordlist.items():
            if value in v:
                ret.append(k)
    return ret


if __name__ == '__main__':
    print("This is a module!")
    exit(1)
