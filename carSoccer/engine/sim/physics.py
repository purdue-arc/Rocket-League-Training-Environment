import math, torch, carSoccer.config as cfg
from torch import Tensor

def ackermannSplit(steer: Tensor, wheelbase: Tensor, track: Tensor) -> tuple[Tensor, Tensor]:
  """Centre steer angle (N, 2P) rad -> (left, right) wheel angles; wheelbase and track are (N, 1) m per world."""
  t = torch.tan(steer)
  return torch.atan(wheelbase * t / (wheelbase - track / 2 * t)), torch.atan(wheelbase * t / (wheelbase + track / 2 * t))

def motorStep(throttle: Tensor, effective: Tensor, game: cfg.GameConfig, dt: float) -> Tensor:
  """Deadband then first-order lag; returns the new effective duty in [-1, 1]. Response curve, voltage sag and
  ESC reverse logic come with system identification (M13)."""
  u = torch.where(throttle.abs() < game.throttleDeadband, torch.zeros_like(throttle), throttle)
  alpha = 1 - math.exp(-dt / game.motorTau)
  return effective + alpha * (u - effective)

def servoStep(target: Tensor, current: Tensor, game: cfg.GameConfig, dt: float) -> Tensor:
  """Steering servo slew limit: move the centre angle toward target by at most steerRate * dt."""
  step = game.steerRate * dt
  return current + (target - current).clamp(-step, step)

def ballAero(vel: Tensor, angvel: Tensor, ball: cfg.BallConfig, game: cfg.GameConfig) -> Tensor:
  """Drag -1/2 rho Cd A |v| v plus Magnus 1/2 rho A r CL (w x v), N, (N, 3)."""
  area = math.pi * ball.radius ** 2
  drag = -0.5 * game.airDensity * game.ballDrag * area * vel.norm(dim=-1, keepdim=True) * vel
  magnus = 0.5 * game.airDensity * area * ball.radius * game.ballLift * torch.cross(angvel, vel, dim=-1)
  return drag + magnus
