from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, x, y):
        """
        :param x: List of tensors of shape (batch_size, *).
        :param y: List of tensors of shape (batch_size, *).
        """

        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
