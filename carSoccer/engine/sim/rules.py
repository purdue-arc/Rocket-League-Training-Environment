import torch, carSoccer.config as cfg
from torch import Tensor
from carSoccer.engine import quat

def ballInGoal(ballPos: Tensor, arena: cfg.ArenaConfig, ballRadius: float) -> Tensor:
  """(N, 2) bool: [ball fully inside orange's goal at +x (blue scores), fully inside blue's goal at -x (orange scores)]."""
  x, y, z = ballPos.unbind(-1)
  mouth = (y.abs() < arena.goalWidth / 2) & (z < arena.goalHeight)
  line = arena.length / 2 + ballRadius
  return torch.stack([mouth & (x > line), mouth & (x < -line)], -1)

def latchGoals(inGoal: Tensor, latch: Tensor) -> tuple[Tensor, Tensor]:
  """A goal fires on the tick the ball first crosses the line, and not again until the latch is cleared at kickoff.
  Returns (goal (N, 2), new latch (N,))."""
  goal = inGoal & ~latch[:, None]
  return goal, latch | inGoal.any(-1)

def ballOut(ballPos: Tensor, arena: cfg.ArenaConfig, margin: float) -> Tensor:
  """(N,) bool: ball left the arena box (goals included) by more than margin, or fell through the floor."""
  x, y, z = ballPos.unbind(-1)
  return (x.abs() > arena.length / 2 + arena.goalDepth + margin) | (y.abs() > arena.width / 2 + margin) | (z < -margin)

def flipped(carQuat: Tensor, threshold: float) -> Tensor:
  """(N, 2P) bool: car on its side or roof."""
  return quat.upZ(carQuat) < threshold
