class FileReader(object):
    def get_sample(self, idx):
        raise NotImplementedError

    def get_range(self, start, end):
        raise NotImplementedError
