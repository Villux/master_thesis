class DataWriter(object):
    def __len__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def write_chunk(self, chunk, labels):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
