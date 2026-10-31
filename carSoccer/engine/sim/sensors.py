import torch, carSoccer.config as cfg
from tensordict import TensorDict
from torch import Tensor
from carSoccer.engine import quat
from carSoccer.engine.base import GameState

# every tracked object packed as pos 3, quat 4, vel 3, angvel 3; cars first, the ball last
POS, QUAT, VEL, ANG = slice(0, 3), slice(3, 7), slice(7, 10), slice(10, 13)

def measure(state: GameState) -> TensorDict:
  """A perfect sensor: the true car and ball state."""
  return TensorDict({"car": state["car"].clone(), "ball": state["ball"].clone()}, batch_size=state.batch_size)

class SensorModel:
  """What the tracking system would report, with its failure modes rather than plain Gaussian noise: timestamp lag
  and jitter, per-object noise (the ball tracked worse than the cars), heavy-tailed position spikes, and dropouts in
  bursts (a two-state Markov chain per object) during which the last measurement is held. Both teams share it."""

  def __init__(self, numWorlds: int, carsPerTeam: int, sensors: cfg.SensorConfig, generator: torch.Generator, device: torch.device):
    self.cfg, self.generator, self.device = sensors, generator, device
    N, C = numWorlds, 2 * carsPerTeam
    self.C = C
    self.history = torch.zeros(N, sensors.maxLagTicks + 1, C + 1, 13, device=device)
    self.ringPos = 0
    self.worldIdx = torch.arange(N, device=device)
    self.dropped = torch.zeros(N, C + 1, dtype=torch.bool, device=device)
    self.last = torch.zeros(N, C + 1, 13, device=device)
    perObject = lambda car, ball: torch.tensor([car] * C + [ball], device=device)
    self.posStd = perObject(sensors.carPosNoise, sensors.ballPosNoise)[:, None]
    self.velStd = perObject(sensors.carVelNoise, sensors.ballVelNoise)[:, None]
    self.dropStart = perObject(sensors.dropStart, sensors.dropStart * sensors.ballDropScale)

  def pack(self, state: GameState) -> Tensor:
    car, ball = state["car"], state["ball"]
    ballQuat = torch.zeros_like(ball["pos"][:, :1]).expand(-1, 4).clone()
    ballQuat[:, 0] = 1
    cars = torch.cat([car["pos"], car["quat"], car["vel"], car["angvel"]], -1)
    balls = torch.cat([ball["pos"], ballQuat, ball["vel"], ball["angvel"]], -1)[:, None]
    return torch.cat([cars, balls], 1)                                       # (N, C + 1, 13)

  def unpack(self, x: Tensor, state: GameState) -> TensorDict:
    C, N = self.C, x.shape[0]
    car = TensorDict({
      "pos": x[:, :C, POS], "quat": x[:, :C, QUAT], "vel": x[:, :C, VEL], "angvel": x[:, :C, ANG],
      "steer": state["car", "steer"].clone(), "uprightTime": state["car", "uprightTime"].clone(),   # on-board, not tracked
    }, batch_size=[N])
    ball = TensorDict({"pos": x[:, C, POS], "vel": x[:, C, VEL], "angvel": x[:, C, ANG]}, batch_size=[N])
    return TensorDict({"car": car, "ball": ball}, batch_size=[N])

  def randn(self, *shape: int) -> Tensor:
    return torch.randn(*shape, generator=self.generator, device=self.device)

  def rand(self, *shape: int) -> Tensor:
    return torch.rand(*shape, generator=self.generator, device=self.device)

  def measure(self, state: GameState) -> TensorDict:
    if self.cfg.isPerfect():                                                 # config, not a tensor: no sync
      return measure(state)
    s, N, C = self.cfg, state.batch_size[0], self.C
    R = self.history.shape[1]
    self.ringPos = (self.ringPos + 1) % R
    self.history[:, self.ringPos] = self.pack(state)
    lag = (s.lagTicks + (self.rand(N) < s.lagJitterProb).long()).clamp(max=R - 1)
    x = self.history[self.worldIdx, (self.ringPos - lag) % R].clone()

    x[..., POS] += self.randn(N, C + 1, 3) * self.posStd
    x[..., VEL] += self.randn(N, C + 1, 3) * self.velStd
    x[..., ANG] += self.randn(N, C + 1, 3) * s.angvelNoise
    spike = self.rand(N, C + 1, 1) < s.spikeProb
    x[..., POS] += torch.where(spike, self.randn(N, C + 1, 3) * s.spikeSize, torch.zeros_like(x[..., POS]))
    x[:, :C, QUAT] = quat.multiply(quat.fromYaw(self.randn(N, C) * s.yawNoise), x[:, :C, QUAT])

    start, end = self.rand(N, C + 1) < self.dropStart, self.rand(N, C + 1) < s.dropEnd
    self.dropped = torch.where(self.dropped, ~end, start)
    self.last = torch.where(self.dropped[..., None], self.last, x)
    return self.unpack(self.last, state)

  def reset(self, mask: Tensor, state: GameState) -> TensorDict:
    """Masked worlds start with a clean track at the true state; returns that measurement for every world."""
    x = self.pack(state)
    self.history = torch.where(mask[:, None, None, None], x[:, None], self.history)
    self.dropped = self.dropped & ~mask[:, None]
    self.last = torch.where(mask[:, None, None], x, self.last)
    return self.unpack(self.last, state)
