import copy
import logging
import random
import torch


def labelFlipping(dataset, indices, poisoned_percent=0, targeted=False, target_label=3, target_changed_label=7):
    """
    select flipping_percent of labels, and change them to random values.
    Args:
        dataset: the dataset of training data, torch.util.data.dataset like.
        indices: Indices of subsets, list like.
        flipping_percent: The ratio of labels want to change, float like.
    """
    new_dataset = copy.deepcopy(dataset)

    targets = torch.tensor(new_dataset.targets).detach().clone()
    class_list = set(torch.tensor(targets).tolist())
    num_indices = len(indices)
    # classes = new_dataset.classes
    # class_to_idx = new_dataset.class_to_idx
    # class_list = [class_to_idx[i] for i in classes]

    if targeted == False:
        print("[labelFlipping]: Untargeted label-flipping!")
        num_flipped = int(poisoned_percent * num_indices)
        if num_indices == 0:
            return new_dataset
        if num_flipped > num_indices:
            return new_dataset
        flipped_indice = random.sample(indices, num_flipped)
        
        for i in flipped_indice:
            t = targets[i]
            flipped = torch.tensor(random.sample(class_list, 1)[0])
            while t == flipped:
                flipped = torch.tensor(random.sample(class_list, 1)[0])
            targets[i] = flipped
    else:
        num_changed = 0
        for i in indices:
            if int(targets[i]) == int(target_label):
                num_changed += 1
                targets[i] = torch.tensor(target_changed_label)
        print("Targeted label-flipping: changed {} labels from {} to {}"
              .format(num_changed, target_label, target_changed_label))
    new_dataset.targets = targets
    return new_dataset
