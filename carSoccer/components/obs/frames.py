import math, torch
from torch import Tensor

def teamFrame(xy: Tensor, sign: Tensor) -> Tensor:
  """Rotate world (..., 2) vectors into a team frame: orange (sign -1) is rotated 180 degrees about z, so every agent
  attacks toward +x. sign broadcasts against xy's leading dims."""
  return xy * sign[..., None]

def teamYaw(yaw: Tensor, sign: Tensor) -> Tensor:
  return torch.where(sign > 0, yaw, torch.remainder(yaw, 2 * math.pi) - math.pi)

def egoFrame(dxy: Tensor, yaw: Tensor) -> Tensor:
  """Rotate (..., 2) vectors into a car's own frame (x forward, y left): (c dx + s dy, -s dx + c dy)."""
  c, s = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
  dx, dy = dxy[..., :1], dxy[..., 1:]
  return torch.cat([c * dx + s * dy, -s * dx + c * dy], -1)
