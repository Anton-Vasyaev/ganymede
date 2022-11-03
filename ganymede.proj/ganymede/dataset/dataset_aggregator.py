# python
from typing      import Any, List
from dataclasses import dataclass
# 3rd party
from nameof import nameof


@dataclass
class DatasetIndex:
    dataset     : Any
    start_idx   : int
    end_idx     : int


class DatasetAggregator:
    def binary_seach_dataset_index(self, search_idx):
        left_idx  = 0
        right_idx = len(self.dataset_indexes)

        while True:
            center_idx = left_idx + (right_idx - left_idx) // 2

            curr_dataset_idx = self.dataset_indexes[center_idx]
            start_idx = curr_dataset_idx.start_idx
            end_idx   = curr_dataset_idx.end_idx
            
            if start_idx <= search_idx and search_idx < end_idx:
                return curr_dataset_idx

            if search_idx > start_idx:
                left_idx = center_idx
            else:
                right_idx = center_idx


    def __init__(
        self,
        datasets
    ):
        self.sum_len = sum([len(d) for d in datasets])

        self.dataset_indexes : List[DatasetIndex] = []

        offset_idx = 0
        for dataset in datasets:
            start_idx = offset_idx
            end_idx   = offset_idx + len(dataset)

            self.dataset_indexes.append(
                DatasetIndex(dataset, start_idx, end_idx)
            )
            
            offset_idx += len(dataset)


    def __len__(self):
        return self.sum_len


    def __getitem__(self, idx):
        if idx < 0 or idx >= self.sum_len:
            raise IndexError(f'{nameof(DatasetAggregator)} index out of range')

        dataset_index = self.binary_seach_dataset_index(idx)

        real_idx = idx - dataset_index.start_idx

        return dataset_index.dataset[real_idx]