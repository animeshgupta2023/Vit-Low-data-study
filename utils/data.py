import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms

def get_cifar10_subset(root='./data', num_per_class=100, seed=42, train_aug=True):
    """
    Returns a Subset of CIFAR-10 with exactly `num_per_class` images per class.
    The transform includes strong augmentation for training, minimal for evaluation.
    """
    np.random.seed(seed)

    # Training augmentations (used for labelled subset and MIM pretraining)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    # Simple evaluation transform (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    # Load the full training dataset once (without transforms)
    full_dataset = CIFAR10(root=root, train=True, download=True, transform=None)
    targets = np.array(full_dataset.targets)

    # Pick exactly num_per_class samples per class
    indices = []
    for cls in range(10):
        cls_idx = np.where(targets == cls)[0]
        np.random.shuffle(cls_idx)
        indices.extend(cls_idx[:num_per_class])

    # Build a Subset – we'll override the transform later
    subset = Subset(full_dataset, indices)

    # Choose the right transform based on mode
    subset.dataset.transform = train_transform if train_aug else eval_transform
    return subset


def get_cifar10_full_test(root='./data'):
    """Returns the full CIFAR-10 test set with the evaluation transform."""
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    return CIFAR10(root=root, train=False, download=True, transform=eval_transform)