from utils.data_writer import DataWriter

class DataWriterOrchestrator(DataWriter):
    def __init__(self, sources):
        self.sources = sources

    def __len__(self):
        return len(self.sources)

    def __str__(self):
        return f"DataWriterOrchestrator for sources: {self.sources}"

    def write_chunk(self, chunks, labels):
        assert len(chunks) == len(labels) == len(self.sources)
        for idx, source in enumerate(self.sources):
            source.write_chunk(chunks[idx], labels[idx])

    def close(self):
        for _, source in enumerate(self.sources):
            source.close()


if __name__ == "__main__":
    # Test
    import os
    import numpy as np
    from data_writer_local import DataWriterLocal

    data = np.array([[1,3,5], [2,4,6]])
    fw_t = DataWriterLocal("train_db")
    fw_v = DataWriterLocal("val_db")
    fw_tt = DataWriterLocal("test_db")

    dw = DataWriterOrchestrator([fw_t, fw_v, fw_tt])

    training_range = range(10)
    training_chunk = np.array([data * i for i in training_range])
    training_label = np.array([np.array([i]) for i in training_range])

    validation_range = range(10, 15)
    validation_chunk = np.array([data * i for i in validation_range])
    validation_label = np.array([np.array([i]) for i in validation_range])

    test_range = range(15, 20)
    test_chunk = np.array([data * i for i in test_range])
    test_label = np.array([np.array([i]) for i in test_range])

    dw.write_chunk(np.array([training_chunk, validation_chunk, test_chunk]), np.array([training_label, validation_label, test_label]))

    assert int(dw.sources[0].dset[9][0][0]) == 9
    assert int(dw.sources[0].dset[9][0][2]) == 45
    assert dw.sources[0].dset_label[9] == 9
    assert dw.sources[0].dset_label[8] == 8

    assert int(dw.sources[1].dset[4][0][0]) == 14
    assert int(dw.sources[1].dset[4][0][2]) == 70
    assert dw.sources[1].dset_label[4] == 14

    assert int(dw.sources[2].dset[4][0][0]) == 19
    assert int(dw.sources[2].dset[4][0][2]) == 95
    assert dw.sources[2].dset_label[4] == 19