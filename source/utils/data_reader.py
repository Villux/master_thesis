class DataReader(object):
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return self.source.dset.shape[0]

    def get_sample(self, idx):
        return self.source[idx], self.source.dset_label[idx]

    def get_range(self, start, end):
        return self.source.dset[start:end], self.source.dset_label[start:end]