import torch, carSoccer.config as cfg
from dataclasses import dataclass
from torch import Tensor

@dataclass
class Ctx:
  """Shared per-step context every component reads instead of recomputing (batched shared_info)."""
  generator: torch.Generator               # the env's one seeded generator
  device: torch.device
  dt: float                                # s per control tick
  arena: cfg.ArenaConfig
  ballRadius: float                        # m
  carsPerTeam: int                         # P
  spawnHeights: Tensor                     # (V,) m, chassis centre height on its wheels, per car variant
  matchTicks: int                          # control ticks per match

  @property
  def numVariants(self) -> int:
    return self.spawnHeights.shape[0]

  def rand(self, *shape: int) -> Tensor:
    return torch.rand(*shape, generator=self.generator, device=self.device)

  def randUniform(self, shape: tuple, lo: float, hi: float) -> Tensor:
    return lo + (hi - lo) * self.rand(*shape)

  def teamSign(self) -> Tensor:
    """(2P,) +1 for blue cars, -1 for orange: the x and y sign of each car's team frame."""
    P = self.carsPerTeam
    return torch.tensor([1.0] * P + [-1.0] * P, device=self.device)
