class DataReader(object):
    def __len__(self):
        raise NotImplementedError
    def get_sample(self, idx):
        raise NotImplementedError

    def get_range(self, start, end):
        raise NotImplementedError
