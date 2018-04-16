import numpy as np
from utils.data_writer import DataWriter

class DataWriterLocal(DataWriter):
    def __init__(self, dataset_name):
        self.name = dataset_name
        self.dset = None
        self.dset_label = None

    def __len__(self):
        return self.dset.shape[0]

    def __str__(self):
        return f"DataWriterLocal for dataset: {self.name}"

    def write_chunk(self, chunk, labels):
        if self.dset is None:
            self.dset = chunk
        else:
            self.dset = np.concatenate((self.dset, chunk))

        if self.dset_label is None:
            self.dset_label = labels
        else:
            self.dset_label = np.concatenate((self.dset_label, labels))

    def close(self):
        # Do not do anything
        pass


if __name__ == "__main__":
    # Test
    import os
    import numpy as np

    data = [[1,3, 5], [2,4, 6]]
    fw = DataWriterLocal("test_db")

    for i in range(10):
        fw.write_chunk(np.array([data]) * i, [i])

    assert int(fw.dset[9][0][0]) == 9
    assert int(fw.dset[8][0][0]) == 8
    assert int(fw.dset[9][0][2]) == 45
    assert fw.dset_label[9] == 9
    assert fw.dset_label[8] == 8
    assert fw.dset_label[8] != 9

    fw.close()
    assert fw.dset is None
    assert fw.dset_label is None
