class Cache(object):
    def get_index(self, idx):
        raise NotImplementedError

    def update_cache(self):
        raise NotImplementedError