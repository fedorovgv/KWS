import logging

import torch
import torch.nn.functional as F
from model import CRNN
from config import TaskConfig
from typing import Optional

logger = logging.getLogger(__file__)


class CRNNStream(CRNN):
    def __init__(
            self,
            config: TaskConfig
    ) -> None:
        super().__init__(config)

        # field storage for torch script
        self._device: str = config.device
        self._num_classes: int = config.num_classes
        self._kernel_size: tuple[int, int] = config.kernel_size

        self._streaming: bool = False
        self._max_window_length: int = config.max_window_length
        self._chunks_buffer_size: int = config.max_window_length * self.config.stride[1]

        self._gru_hidden: torch.Tensor = torch.zeros((
            config.gru_num_layers,
            1,
            config.hidden_size,
        ))
        self._chunks_buffer: torch.Tensor = torch.tensor([], device=self._device)
        self._gru_output_buffer: torch.Tensor = torch.tensor([], device=self._device)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        if not self._streaming:
            batch = chunk.unsqueeze(dim=1)
            conv_output = self.conv(batch).transpose(-1, -2)
            gru_output, _ = self.gru(conv_output)
            contex_vector = self.attention(gru_output)
            output = self.classifier(contex_vector)
            return output
        else:
            chunk = chunk.unsqueeze(dim=1)
            self._chunks_buffer = torch.cat([self._chunks_buffer, chunk], dim=-1)
            self._chunks_buffer = self._chunks_buffer[:, :, :, -self._chunks_buffer_size:]

            if self._chunks_buffer.size(-1) > self._kernel_size[1]:
                conv_output = self.conv(self._chunks_buffer).transpose(-1, -2)
            else:
                return torch.zeros((1, self._num_classes))

            gru_output, self._gru_hidden = self.gru(conv_output, self._gru_hidden)
            self._gru_output_buffer = torch.cat([self._gru_output_buffer, gru_output], dim=1)
            self._gru_output_buffer = self._gru_output_buffer[:, -self._max_window_length:, :]

            contex_vector = self.attention(self._gru_output_buffer)
            output = self.classifier(contex_vector)
            return output

    @torch.no_grad()
    def inference(self, chunk: torch.Tensor) -> torch.Tensor:
        if not self._streaming:
            self.streaming = True
        logits = self(chunk)
        probs = F.softmax(logits, dim=-1).detach().cpu()
        probs = probs[0].item()
        return probs

    @property
    def streaming(self):
        return self._streaming

    @streaming.setter
    def streaming(self, value: bool) -> None:
        if not value:
            self._clean_buffers()
        self._streaming = value

    def _clean_buffers(self) -> None:
        self._chunks_buffer = torch.tensor([], device=self._device)
        self._gru_output_buffer = torch.tensor([], device=self._device)
