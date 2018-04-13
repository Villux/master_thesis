import h5py

class FileWriter(object):
    def __init__(self, dataset_name, H, W):
        self.path = f"data/{dataset_name}.h5"
        self.file = h5py.File(self.path, 'w')
        self.row_count = 0
        max_shape = (None, H, W)
        self.dset = self.file.create_dataset('data', shape=(0, H, W), maxshape=max_shape)

    def write_chunk(self, chunk):
        self.dset.resize(self.row_count + chunk.shape[0], axis=0)
        self.dset[self.row_count:] = chunk
        self.row_count += chunk.shape[0]

    def close_file(self):
        self.file.close()

if __name__ == "__main__":
    # Test
    import os
    import numpy as np

    data = [[1,3, 5], [2,4, 6]]
    fw = FileWriter("test_db", 2, 3)

    for i in range(10):
        fw.write_chunk(np.array([data]) * i)

    assert int(fw.dset[9][0][0]) == 9
    assert int(fw.dset[8][0][0]) == 8
    assert int(fw.dset[9][0][2]) == 45

    fw.close_file()
    os.remove(fw.path)
