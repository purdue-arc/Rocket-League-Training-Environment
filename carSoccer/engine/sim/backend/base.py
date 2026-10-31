import torch
from typing import Protocol
from torch import Tensor

class SimBackend(Protocol):
  """The only code that touches MuJoCo during training. Everything above reads qpos, qvel and sensordata."""
  numWorlds: int
  device: torch.device
  qpos: Tensor          # (N, nq)  live views, updated by step()
  qvel: Tensor          # (N, nv)
  qpos0: Tensor         # (nq,)    the model's reference pose, for resets
  sensordata: Tensor    # (N, nsensor) peak value over the last step's substeps, e.g. ball-contact flags

  def step(self, ctrl: Tensor, xfrc: Tensor | None, nsub: int) -> None:
    """ctrl (N, nu) and xfrc (N, nbody, 6) are held constant for nsub substeps."""
  def writeState(self, mask: Tensor, qpos: Tensor, qvel: Tensor) -> None:
    """Overwrite worlds where mask is True (resets, kickoffs, flip recovery)."""
  def setParams(self, mask: Tensor, params: dict[str, Tensor]) -> None:
    """Per-world model fields (friction, mass, gains, pair solref, a variant's ParamPack)."""
