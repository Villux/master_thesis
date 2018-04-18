import h5py
from utils.data_reader import DataReader

class DataReaderH5py(DataReader):
    def __init__(self, dataset_name):
        self.path = f"data/{dataset_name}.h5"
        self.file = h5py.File(self.path, 'r')

        self.dset = self.file['data']
        self.dset_label = self.file['label']

    def __len__(self):
        return len(self.dset)

    def get_sample(self, idx):
        return self.dset[idx], self.dset_label[idx]

    def get_range(self, start, end):
        return self.dset[start:end], self.dset_label[start:end]
