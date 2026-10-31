import torch
from torch import Tensor, nn

class RunningNorm(nn.Module):
  """Running mean and variance per feature, updated during collection only. Buffers, so they are saved in the
  checkpoint and exported with the actor."""

  def __init__(self, size: int, clip: float = 5.0):
    super().__init__()
    self.clip = clip
    self.register_buffer("mean", torch.zeros(size))
    self.register_buffer("var", torch.ones(size))
    self.register_buffer("count", torch.tensor(1e-4))

  @torch.no_grad()
  def update(self, x: Tensor) -> None:
    x = x.reshape(-1, x.shape[-1])
    n = x.shape[0]
    mean, var = x.mean(0), x.var(0, unbiased=False)
    delta, total = mean - self.mean, self.count + n
    self.mean += delta * n / total
    self.var = (self.var * self.count + var * n + delta ** 2 * self.count * n / total) / total
    self.count = total

  def forward(self, x: Tensor) -> Tensor:
    return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)

class RunningStd:
  """Running variance of a scalar stream (the discounted return), for reward normalisation."""

  def __init__(self):
    self.mean, self.var, self.count = 0.0, 1.0, 1e-4

  def update(self, x: Tensor) -> None:
    n = x.numel()
    mean, var = x.mean().item(), x.var(unbiased=False).item()
    delta, total = mean - self.mean, self.count + n
    self.mean += delta * n / total
    self.var = (self.var * self.count + var * n + delta ** 2 * self.count * n / total) / total
    self.count = total

  @property
  def std(self) -> float:
    return (self.var + 1e-8) ** 0.5
