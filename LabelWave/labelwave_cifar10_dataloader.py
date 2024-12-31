import numpy as np
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from noise_build import dataset_split


class CifarDataset(Dataset):
    def __init__(self, data, labels, paths, transform=None):
        self.data = data
        self.labels = labels
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        img, label, path = self.data[index], self.labels[index], self.paths[index]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label, path

    def __len__(self):
        return len(self.data)


class CifarDataLoader:
    def __init__(self, root_dir, noise_mode, r, batch_size, num_workers, random_seed=1):
        self.root_dir = root_dir
        self.noise_mode = noise_mode
        self.r = r
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_data(self):
        train_data = []
        train_labels = []
        train_paths = []

        for i in range(1, 6):
            data_dic = self.unpickle(os.path.join(self.root_dir, f"data_batch_{i}"))
            train_data.append(data_dic[b'data'])
            train_labels += data_dic[b'labels']
            train_paths += [f'data_batch_{i}_{j}' for j in range(len(data_dic[b'labels']))]

        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))

        test_data_dic = self.unpickle(os.path.join(self.root_dir, "test_batch"))
        test_data = test_data_dic[b'data']
        test_data = test_data.reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
        test_labels = test_data_dic[b'labels']
        test_paths = [f'test_batch_{i}' for i in range(len(test_labels))]

        noise_labels = dataset_split(train_images=train_data,
                                     train_labels=train_labels,
                                     noise_rate=self.r,
                                     noise_type=self.noise_mode,
                                     random_seed=self.random_seed,
                                     num_classes=10)

        num_samples = train_data.shape[0]
        np.random.seed(self.random_seed)
        train_set_index = np.random.choice(num_samples, int(num_samples * 1.0), replace=False)
        index = np.arange(num_samples)

        train_set = train_data
        train_labels = noise_labels
        train_paths = np.array(train_paths)

        return train_set, train_labels, train_paths, test_data, test_labels, test_paths

    def run(self, mode):
        train_set, train_labels, train_paths, test_data, test_labels, test_paths = self.load_data()

        if mode == 'train':
            dataset = CifarDataset(train_set, train_labels, train_paths, transform=self.transform_train)
            dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                    shuffle=True, num_workers=self.num_workers)
        elif mode == 'test':
            dataset = CifarDataset(test_data, test_labels, test_paths, transform=self.transform_test)
            dataloader = DataLoader(dataset, batch_size=1000,
                                    shuffle=False, num_workers=self.num_workers)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return dataloader
