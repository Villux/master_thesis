class DataWriter(object):
    def __init__(self, fw_train, fw_validation, fw_test):
        self.training = fw_train
        self.validation = fw_validation
        self.test = fw_test

    def write_chunk_training(self, *args, **kwargs):
        self.training.write_chunk(*args, **kwargs)

    def write_chunk_validation(self, *args, **kwargs):
        self.validation.write_chunk(*args, **kwargs)

    def write_chunk_test(self, *args, **kwargs):
        self.test.write_chunk(*args, **kwargs)

    def close_writers(self):
        self.training.close_file()
        self.validation.close_file()
        self.test.close_file()


if __name__ == "__main__":
    # Test
    import os
    import numpy as np
    from file_writer_local import FileWriterLocal

    data = [[1,3,5], [2,4,6]]
    fw_t = FileWriterLocal("train_db")
    fw_v = FileWriterLocal("val_db")
    fw_tt = FileWriterLocal("test_db")

    dw = DataWriter(fw_t, fw_v, fw_tt)

    for i in range(10):
        dw.write_chunk_training(np.array([data]) * i, [i])
    for i in range(10, 15):
        dw.write_chunk_validation(np.array([data]) * i, [i])
    for i in range(15, 20):
        dw.write_chunk_test(np.array([data]) * i, [i])

    assert int(dw.training.dset[9][0][0]) == 9
    assert int(dw.training.dset[9][0][2]) == 45
    assert dw.training.dset_label[9] == 9
    assert dw.training.dset_label[8] == 8

    assert int(dw.validation.dset[4][0][0]) == 14
    assert int(dw.validation.dset[4][0][2]) == 70
    assert dw.validation.dset_label[4] == 14

    assert int(dw.test.dset[4][0][0]) == 19
    assert int(dw.test.dset[4][0][2]) == 95
    assert dw.test.dset_label[4] == 19

    dw.close_writers()
    assert dw.training.dset is None
    assert dw.validation.dset is None
    assert dw.test.dset is None