import collections

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from dataset import SpeechCommandDataset
from config import TaskConfig
from augmentation import AugsCreation
from sampler import get_sampler
from collator import Collator
from features import LogMelspec
from model import CRNN
from trainer import train_epoch, validation


# make dataset
dataset = SpeechCommandDataset(
    path2dir='speech_commands', keywords=TaskConfig.keyword
)

indexes = torch.randperm(len(dataset))
train_indexes = indexes[:int(len(dataset) * 0.8)]
val_indexes = indexes[int(len(dataset) * 0.8):]

train_df = dataset.csv.iloc[train_indexes].reset_index(drop=True)
val_df = dataset.csv.iloc[val_indexes].reset_index(drop=True)

train_set = SpeechCommandDataset(csv=train_df, transform=AugsCreation())
val_set = SpeechCommandDataset(csv=val_df)

# make sampler
train_sampler = get_sampler(train_set.csv['label'].values)

#make dataloaders
train_loader = DataLoader(
    train_set, batch_size=TaskConfig.batch_size,
    shuffle=False, collate_fn=Collator(),
    sampler=train_sampler,
    num_workers=2, pin_memory=True,
)

val_loader = DataLoader(
    val_set, batch_size=TaskConfig.batch_size,
    shuffle=False, collate_fn=Collator(),
    num_workers=2, pin_memory=True,
)

# mel spec's
melspec_train = LogMelspec(is_train=True, config=TaskConfig)
melspec_val = LogMelspec(is_train=False, config=TaskConfig)

history = collections.defaultdict(list)

# train
config = TaskConfig(hidden_size=32)
model = CRNN(config).to(config.device)
opt = torch.optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)

for n in range(TaskConfig.num_epochs):
    train_epoch(model, opt, train_loader, melspec_train, config.device)
    au_fa_fr = validation(model, val_loader, melspec_val, config.device)
    history['val_metric'].append(au_fa_fr)

    # clear_output()
    plt.plot(history['val_metric'])
    plt.ylabel('Metric')
    plt.xlabel('Epoch')
    plt.grid()
    plt.show()

    print('END OF EPOCH', n)
