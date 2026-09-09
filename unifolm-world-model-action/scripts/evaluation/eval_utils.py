import torch
import torchvision

from typing import Any, Deque, Dict, Mapping, Union
from torch.utils.tensorboard.writer import SummaryWriter


def populate_queues(
        queues: Dict[str, Deque[Any]],
        batch: Mapping[str, Any]) -> Dict[str, Deque[Any]]:

    for key in batch:
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            queues[key].append(batch[key])
    return queues


def log_to_tensorboard(
        writer: SummaryWriter,
        data: Union[torch.Tensor, Any],
        tag: str,
        fps: int = 10) -> None:
    if isinstance(data, torch.Tensor) and data.dim() == 5:
        video = data
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4)
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video]
        grid = torch.stack(frame_grids, dim=0)
        grid = (grid + 1.0) / 2.0
        grid = grid.unsqueeze(dim=0)
        writer.add_video(tag, grid, fps=fps)
