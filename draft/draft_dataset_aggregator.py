import dependencies
# python
import random
# project
from ganymede.dataset.dataset_aggregator import DatasetAggregator


class DatasetPlaceholder:
    def __init__(self, start_idx, len):
        self.start_idx = start_idx
        self.len = len

    
    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        if idx >= self.len or idx < 0:
            raise IndexError(f'invalid idx:{idx}')

        return self.start_idx + idx
        

def draft_dataset_aggregator():
    datasets = []

    max_len = 512

    random_i = random.Random(1024)

    offset_idx = 0
    for i in range(15):
        random_len = int(random_i.random() * max_len)

        dataset = DatasetPlaceholder(offset_idx, random_len)

        datasets.append(dataset)

        offset_idx += random_len

    aggregator = DatasetAggregator(datasets)

    for idx in range(len(aggregator)):
        data_idx = aggregator[idx]

        if data_idx != idx:
            print(f'invalid:{idx} != {data_idx}')
            break
        

if __name__ == '__main__':
    draft_dataset_aggregator()