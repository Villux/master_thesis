import h5py

class FileWriter(object):
    def __init__(self, dataset_name, H, W):
        self.path = f"data/{dataset_name}.h5"
        self.file = h5py.File(self.path, 'w')
        self.row_count = 0
        self.dset = self.file.create_dataset('data', shape=(0, H, W), maxshape=(None, H, W))
        self.dset_label = self.file.create_dataset('label', shape=(0,), maxshape=(None,), dtype='i8')

    def write_chunk(self, chunk, labels):
        self.dset.resize(self.row_count + chunk.shape[0], axis=0)
        self.dset[self.row_count:] = chunk

        self.dset_label.resize(self.row_count + chunk.shape[0], axis=0)
        self.dset_label[self.row_count:] = labels
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
        fw.write_chunk(np.array([data]) * i, i)

    assert int(fw.dset[9][0][0]) == 9
    assert int(fw.dset[8][0][0]) == 8
    assert int(fw.dset[9][0][2]) == 45
    assert fw.dset_label[9] == 9
    assert fw.dset_label[8] == 8
    assert fw.dset_label[8] != 9

    import ipdb; ipdb.set_trace()

    fw.close_file()
    os.remove(fw.path)
