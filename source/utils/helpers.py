from utils.batch_cache import BatchCache
from utils.file_reader_h5py import FileReaderH5py
from utils.h5py_dataset import H5pyDataset

def create_h5py_dataset_with_cache(dataset_name, batch_size):
    file_reader = FileReaderH5py(dataset_name)
    cache = BatchCache(batch_size, file_reader.get_range)
    return H5pyDataset(file_reader, cache=cache)

