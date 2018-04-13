import h5py
from torch.utils.data import Dataset

class FileReader(object):
    def __init__(self, dataset_name):
        self.path = f"data/{dataset_name}.h5"
        self.file = h5py.File(self.path, 'r')

        self.dset = self.file['data']
        self.dset_label = self.file['label']

    def get_sample(self, idx):
        return self.dset[idx], self.dset_label[idx]

    def get_range(self, start, end):
        return self.dset[start:end], self.dset_label[start:end]

class Cache(object):
    def __init__(self, batch_size, get_data):
        self.start = 0
        self.end = batch_size
        self.batch_size = batch_size
        self.get_data = get_data
        self.data, self.label = get_data(self.start, self.end)

    def get_from_cache(self, idx):
        if idx >= self.end:
            self.update_cache()

        cache_idx = idx % self.batch_size
        return self.data[cache_idx], self.label[cache_idx]

    def update_cache(self):
        self.data, self.label = self.get_data(self.end, self.end + self.batch_size)



class H5pyDataset(Dataset):
    def __init__(self, dataset_name, batch_size):
        self.path = f"data/{dataset_name}.h5"
        self.fw = FileReader(dataset_name)
        self.batch_size = batch_size
        self.cache = Cache(batch_size, self.fw.get_range)

    def __len__(self):
        return self.fw.dset.shape[0]

    def __getitem__(self, idx):
        data, label = self.cache.get_from_cache(idx)
        return {'data': data, 'label': label[0]}

