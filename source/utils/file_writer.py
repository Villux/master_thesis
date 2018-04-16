class FileWriter(object):
    def __len__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def write_chunk(self, chunk, labels):
        raise NotImplementedError

    def close_file(self):
        raise NotImplementedError
