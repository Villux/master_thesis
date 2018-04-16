from torch.utils.data import Dataset

def create_with_cache(cache, data_reader):
    return H5pyDataset(data_reader, cache)

class H5pyDataset(Dataset):
    def __init__(self, data_reader, cache=None):
        self.cache = cache
        self.data_reader = data_reader

    def __len__(self):
        return len(self.data_reader)

    def __getitem__(self, idx):
        if self.cache:
            data, label = self.cache.get_index(idx)
        else:
            data, label = self.data_reader.get_sample(idx)
        return {'data': data, 'label': label[0]}