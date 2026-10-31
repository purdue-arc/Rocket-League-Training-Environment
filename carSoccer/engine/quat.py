import torch
from torch import Tensor

# quaternions are (w, x, y, z), MuJoCo's order; every function broadcasts over leading dims

def rotate(q: Tensor, v: Tensor) -> Tensor:
  """Rotate vectors v (..., 3) by unit quaternions q (..., 4): body frame -> world frame."""
  w, u = q[..., :1], q[..., 1:]
  t = 2 * torch.cross(u, v, dim=-1)
  return v + w * t + torch.cross(u, t, dim=-1)

def rotateInverse(q: Tensor, v: Tensor) -> Tensor:
  """World frame -> body frame."""
  return rotate(q * q.new_tensor([1, -1, -1, -1]), v)

def yaw(q: Tensor) -> Tensor:
  """Heading about +z, rad, (...,)."""
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def upZ(q: Tensor) -> Tensor:
  """World z component of the body's up axis: 1 upright, 0 on its side, -1 on its roof."""
  x, y = q[..., 1], q[..., 2]
  return 1 - 2 * (x * x + y * y)

def fromYaw(angle: Tensor) -> Tensor:
  """Upright quaternion with the given heading, (..., 4)."""
  half = angle / 2
  zero = torch.zeros_like(half)
  return torch.stack([half.cos(), zero, zero, half.sin()], -1)

def multiply(a: Tensor, b: Tensor) -> Tensor:
  """Hamilton product a * b: rotate by b, then by a."""
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ], -1)
