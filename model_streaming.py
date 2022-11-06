import torch

from model import CRNN
from config import TaskConfig
from typing import Optional


class CRNNStream(CRNN):
    def __init__(
            self,
            config: TaskConfig,
            max_window_length: int,
            streaming: bool = False,
    ) -> None:
        super(CRNNStream, self).__init__(config)

        self._streaming: bool = streaming
        self._max_window_length = max_window_length

        self._features_buffer: Optional[torch.Tensor] = None
        self._gru_output_buffer: Optional[torch.Tensor] = None


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._streaming:
            return super(CRNN, self).forward(input)
        else:
            input = input.unsqueeze(dim=1)


    @property
    def streaming(self):
        return self._streaming

    @streaming.setter
    def streaming(self, value: bool) -> None:
        if value:
            self._init_buffers()
        else:
            self._clean_buffers()
        self._streaming = value

    def _init_buffers(self) -> None:
        self._features_buffer = torch.Tensor([])
        self._gru_output_buffer = torch.Tensor([])

    def _clean_buffers(self) -> None:
        self._features_buffer = None
        self._gru_output_buffer = None
