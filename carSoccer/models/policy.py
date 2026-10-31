import math, torch
from dataclasses import dataclass
from tensordict import TensorDict
from torch import Tensor, nn
from carSoccer.components.obs.builders import ObsSpec
from carSoccer.models.encoders import ENCODERS, actorSize, flattenObs, mlp, orthogonal
from carSoccer.models.normalizer import RunningNorm

@dataclass(frozen=True)
class ModelSpec:
  encoder: str = "flatMlp"                 # registered encoder name
  hidden: tuple[int, ...] = (256, 256)     # actor encoder layers
  criticHidden: tuple[int, ...] = (256, 256)
  initStd: float = 0.6                     # initial action std, in action units
  throttleBias: float = 0.3                # initial throttle mean, so early exploration actually moves the cars
  logStdRange: tuple[float, float] = (-3.0, 0.5)

@dataclass
class ModelOut:
  dist: torch.distributions.Normal         # over raw [throttle, steer]; the env clamps to [-1, 1]
  value: Tensor                            # (B,)

class AgentModel(nn.Module):
  """One car's policy with shared weights across slots (car_shared paradigm): normaliser -> encoder -> Gaussian head.
  The critic is a separate MLP on the privileged, noise-free `critic` entry."""

  def __init__(self, obsSpec: ObsSpec, numSlots: int, spec: ModelSpec = ModelSpec()):
    super().__init__()
    self.spec, self.numSlots = spec, numSlots
    actorIn, criticIn = actorSize(obsSpec, numSlots), obsSpec.critic + numSlots
    self.actorNorm, self.criticNorm = RunningNorm(actorIn), RunningNorm(criticIn)
    self.encoder = ENCODERS[spec.encoder](actorIn, spec.hidden)
    self.mean = orthogonal(nn.Linear(self.encoder.outSize, 2), 0.01)
    with torch.no_grad():
      self.mean.bias[0] = spec.throttleBias
    self.logStd = nn.Parameter(torch.full((2,), math.log(spec.initStd)))
    self.critic = nn.Sequential(mlp(criticIn, spec.criticHidden), orthogonal(nn.Linear(spec.criticHidden[-1], 1), 1.0))

  def actorInput(self, obs: TensorDict) -> Tensor:
    return flattenObs(obs, self.numSlots)

  def criticInput(self, obs: TensorDict) -> Tensor:
    return flattenObs(obs, self.numSlots, keys=("critic",))

  def updateNormalizers(self, obs: TensorDict) -> None:
    self.actorNorm.update(self.actorInput(obs))
    self.criticNorm.update(self.criticInput(obs))

  def actionMean(self, actorIn: Tensor) -> Tensor:
    return self.mean(self.encoder(self.actorNorm(actorIn)))

  def value(self, obs: TensorDict) -> Tensor:
    return self.critic(self.criticNorm(self.criticInput(obs))).squeeze(-1)

  def forward(self, obs: TensorDict) -> ModelOut:
    mean = self.actionMean(self.actorInput(obs))
    std = self.logStd.clamp(*self.spec.logStdRange).exp().expand_as(mean)
    return ModelOut(dist=torch.distributions.Normal(mean, std), value=self.value(obs))

  @torch.no_grad()
  def act(self, obs: TensorDict, deterministic: bool = True) -> Tensor:
    """(B, 2) actions for playing, not training."""
    mean = self.actionMean(self.actorInput(obs))
    if deterministic:
      return mean
    return torch.distributions.Normal(mean, self.logStd.clamp(*self.spec.logStdRange).exp()).sample()

  def exportActor(self, path: str) -> None:
    """Deployment: normaliser + encoder + mean head as TorchScript, taking flattenObs(obs) (B, F) -> (B, 2) in [-1, 1]."""
    actor = Actor(self.actorNorm, self.encoder, self.mean).eval()
    torch.jit.script(actor).save(path)

class Actor(nn.Module):
  def __init__(self, norm: RunningNorm, encoder: nn.Module, mean: nn.Linear):
    super().__init__()
    self.norm, self.encoder, self.mean = norm, encoder, mean

  def forward(self, x: Tensor) -> Tensor:
    return self.mean(self.encoder(self.norm(x))).clamp(-1.0, 1.0)
