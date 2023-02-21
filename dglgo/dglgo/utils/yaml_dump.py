from ruamel.yaml.comments import CommentedMap


def deep_convert_dict(layer):
    to_ret = layer
    if isinstance(layer, dict):
        to_ret = CommentedMap(layer)
    try:
        for key, value in to_ret.items():
            to_ret[key] = deep_convert_dict(value)
    except AttributeError:
        pass

    return to_ret


import collections.abc


def merge_comment(d, comment_dict, column=30):
    for k, v in comment_dict.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = merge_comment(d.get(k, CommentedMap()), v)
        else:
            d.yaml_add_eol_comment(v, key=k, column=column)
    return d
