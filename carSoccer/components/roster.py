import torch
from torch import Tensor

class Roster:
  """Who controls each team, by fixed slices of worlds set at startup. A slice is (fraction, blue controller, orange
  controller); controllers are names such as "learner", "scripted" or "random". Changing the mix means building a
  new Roster, outside the step (a curriculum stage boundary)."""

  def __init__(self, numWorlds: int, carsPerTeam: int, slices: list[tuple[float, str, str]], device: torch.device):
    total = sum(f for f, _, _ in slices)
    counts = [round(numWorlds * f / total) for f, _, _ in slices]
    counts[-1] = numWorlds - sum(counts[:-1])                                # rounding remainder goes to the last slice
    if min(counts) < 0:
      raise ValueError(f"slices {slices} do not fit in {numWorlds} worlds")

    self.names = sorted({name for _, blue, orange in slices for name in (blue, orange)})
    rows = [[self.names.index(blue), self.names.index(orange)] for (_, blue, orange), n in zip(slices, counts) for _ in range(n)]
    team = torch.arange(2 * carsPerTeam, device=device) // carsPerTeam
    self.teamController = torch.tensor(rows, device=device)                  # (N, 2)
    self.carController = self.teamController[:, team]                         # (N, 2P)

  def cars(self, name: str) -> Tensor:
    """(N, 2P) bool: cars this controller drives."""
    return self.carController == self.names.index(name)

  def combine(self, actions: dict[str, Tensor]) -> Tensor:
    """Each controller's (N, 2P, 2) actions -> one carActions tensor, each car taking its own controller's command."""
    out = torch.zeros_like(next(iter(actions.values())))
    for name, a in actions.items():
      out = torch.where(self.cars(name)[..., None], a, out)
    return out
