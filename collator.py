import torch
from torch.nn.utils.rnn import pad_sequence


class Collator:
    def __call__(self, data):
        wavs = [el['wav'] for el in data]
        labels = [el['label'] for el in data]

        wavs = pad_sequence(wavs, batch_first=True)
        labels = torch.Tensor(labels).long()

        return wavs, labels
