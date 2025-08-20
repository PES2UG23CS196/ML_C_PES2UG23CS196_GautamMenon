# lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor) -> float:
    """
    Calculate the entropy of the entire dataset.
    """
    target = tensor[:, -1]  # last column = target
    values, counts = torch.unique(target, return_counts=True)
    probs = counts.float() / counts.sum()

    entropy = -torch.sum(probs * torch.log2(probs))
    return float(entropy)


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of an attribute.
    """
    target = tensor[:, -1]
    attr_values = tensor[:, attribute]

    values, counts = torch.unique(attr_values, return_counts=True)
    total = len(attr_values)

    avg_info = 0.0
    for v, count in zip(values, counts):
        subset = tensor[attr_values == v]
        subset_entropy = get_entropy_of_dataset(subset)
        weight = count.item() / total
        avg_info += weight * subset_entropy

    return float(avg_info)


def get_information_gain(tensor: torch.Tensor, attribute: int) -> float:
    """
    Calculate Information Gain for an attribute.
    """
    total_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)

    info_gain = total_entropy - avg_info
    return round(float(info_gain), 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.
    Returns (dict of gains, best_attribute_index).
    """
    n_attributes = tensor.shape[1] - 1  # exclude target column
    gains = {}

    for attr in range(n_attributes):
        gains[attr] = get_information_gain(tensor, attr)

    # get attribute with max info gain
    best_attr = max(gains, key=gains.get)
    return gains, best_attr
