from utils.cache import Cache

class BatchCache(Cache):
    def __init__(self, batch_size, get_data_fn):
        self.start = 0
        self.end = batch_size
        self.batch_size = batch_size
        self.get_data_fn = get_data_fn
        self.data, self.label = get_data_fn(self.start, self.end)

    def get_index(self, idx):
        if idx >= self.end:
            self.update_cache()

        cache_idx = idx % self.batch_size
        return self.data[cache_idx], self.label[cache_idx]

    def update_cache(self):
        self.data, self.label = self.get_data_fn(self.end, self.end + self.batch_size)