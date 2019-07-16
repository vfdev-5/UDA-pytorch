from torchvision import datasets


def get_train_test_datasets(dataset_name, path):

    assert dataset_name in datasets.__dict__, "Unknown dataset name {}".format(dataset_name)
    fn = datasets.__dict__[dataset_name]

    train_ds = fn(root=path, train=True, download=True)
    test_ds = fn(root=path, train=False, download=False)

    return train_ds, test_ds, len(train_ds.classes)
