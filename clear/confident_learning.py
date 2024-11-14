import pandas as pd
from cleanlab.dataset import find_overlapping_classes




def merge_label(y, merged_dict):
    """
    Merge labels in list `y` based on the provided mapping dictionary with `merged_dict`.

    Args:
        y (list): The list of labels to be merged.
        merged_dict (dict): A dictionary mapping original labels to merged labels.

    Returns:
        list: The list of merged labels.

    """
    y_merged = []
    for x in y:
        if x in merged_dict:
            y_merged.append(merged_dict[x])
        else:
            y_merged.append(x)
    return y_merged
         

def merge_overlap_classes(overlapped, thresh=0.02):
    """
    Merge overlapping classes based on a given threshold.

    Parameters:
    - overlapped (DataFrame): DataFrame containing the overlapped classes.
    - thresh (float): Threshold value for joint probability. Default is 0.02.

    Returns:
    - merged (list): List of sets containing merged classes.
    - merged_dict (dict): Dictionary mapping original classes to merged classes.
    """


    overlap_classes = overlapped[overlapped['Joint Probability']>=thresh]

    classA = 'Class Name A'
    classB = 'Class Name B'
    overlap_classes[classA]=overlap_classes[classA].astype(int)
    merged = []
    merged_dict = {}
    for idx, row in overlap_classes.iterrows():
        ca, cb = int(row[classA]), int(row[classB])
        union = False
        for group in merged:
            if (ca in group) or (cb in group):
                group.add(ca)
                group.add(cb)
                union = True
                break
        if not union:
            merged.append(set([ca, cb]))    
    for group in merged:
        target = min(group)
        for x in group:
            merged_dict[x] = target

    return merged, merged_dict

def find_merged_classes(labels, class_names, pred_probs, thresh=0.02):
    """
    Finds the merged classes based on the given labels, class names, predicted probabilities, and threshold.

    Args:
        labels (list): The labels of the classes.
        class_names (list): The names of the classes.
        pred_probs (list): The predicted probabilities for each class.
        thresh (float, optional): The threshold value for merging classes. Defaults to 0.02.

    Returns:
        tuple: A tuple containing the following:
            - overlapped (list): The list of overlapping classes.
            - merged (list): The list of merged classes.
            - merged_dict (dict): A dictionary mapping the merged classes to their original classes.
    """
    overlapped = find_overlapping_classes(
        labels = labels,
        class_names = class_names,
    #     pred_probs = pred_probs_mean,
        pred_probs = pred_probs
    #     confident_joint = confident_joint
    )
    merged, merged_dict = merge_overlap_classes(overlapped, thresh=thresh)
    return overlapped, merged, merged_dict






