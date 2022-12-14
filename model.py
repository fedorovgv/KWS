import torch
import torch.nn as nn

from config import TaskConfig


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.energy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        energy = self.energy(batch)
        alpha = torch.softmax(energy, dim=-2)
        return (batch * alpha).sum(dim=-2)


class CRNN(nn.Module):
    def __init__(self, config: TaskConfig):
        super().__init__()
        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=config.cnn_out_channels,
                kernel_size=config.kernel_size, stride=config.stride
            ),
            nn.Flatten(start_dim=1, end_dim=2),
        )
        self.conv_out_frequency = self._get_conv_out_frequency(config)

        self.gru = nn.GRU(
            input_size=self.conv_out_frequency * config.cnn_out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.gru_num_layers,
            dropout=0.1,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        self.attention = Attention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    @staticmethod
    def _get_conv_out_frequency(config: TaskConfig) -> int:
        conv_out_frequency = config.n_mels - config.kernel_size[0]
        conv_out_frequency //= config.stride[0]
        return conv_out_frequency + 1

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.unsqueeze(dim=1)
        conv_output = self.conv(batch).transpose(-1, -2)
        gru_output, _ = self.gru(conv_output)
        contex_vector = self.attention(gru_output)
        output = self.classifier(contex_vector)
        return output
