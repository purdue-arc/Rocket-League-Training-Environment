import torch
from tensordict import TensorDict
from carSoccer.components.bots import RandomDriver, chaseBall
from carSoccer.env import SoccerEnv
from carSoccer.learn.runner import loadPolicy

class Controller:
  """Drives every car of a world batch; the caller keeps only the team it controls. Called once per env.step."""

  def __call__(self, state: TensorDict, obs: TensorDict | None) -> torch.Tensor:
    raise NotImplementedError

class PolicyController(Controller):
  """A trained checkpoint, deterministic. With hold > 1 (an env stepping one tick at a time) it keeps each decision
  for `hold` calls, exactly like the action repeat it was trained with."""

  def __init__(self, path: str, env: SoccerEnv, hold: int = 1):
    self.model, self.env, self.hold, self.calls, self.action = loadPolicy(path, env), env, hold, 0, None

  def __call__(self, state, obs):
    if self.calls % self.hold == 0:
      self.action = self.model.act(obs).view(self.env.numWorlds, -1, 2)
    self.calls += 1
    return self.action

class FunctionController(Controller):
  def __init__(self, fn):
    self.fn = fn

  def __call__(self, state, obs):
    return self.fn(state)

def makeController(name: str, env: SoccerEnv, hold: int = 1) -> Controller:
  """"random", "chase", "idle", or a checkpoint path."""
  if name == "random":
    driver = RandomDriver(env.numWorlds, env.ctx.carsPerTeam, env.ctx, redraw=0.08 / hold)
    return FunctionController(driver)
  if name == "chase":
    return FunctionController(lambda state: chaseBall(state, env.ctx))
  if name == "idle":
    return FunctionController(lambda state: torch.zeros(env.numWorlds, 2 * env.ctx.carsPerTeam, 2))
  return PolicyController(name, env, hold)

def teamActions(blue: torch.Tensor, orange: torch.Tensor, carsPerTeam: int) -> torch.Tensor:
  """Blue's actions for blue cars and orange's for orange cars, (N, 2P, 2)."""
  return torch.cat([blue[:, :carsPerTeam], orange[:, carsPerTeam:]], 1)

@torch.no_grad()
def evaluate(env: SoccerEnv, blue: Controller, orange: Controller, matches: int) -> dict[str, float]:
  """Play until `matches` matches finish across the env's worlds (fixed seed, deterministic policies). Blue's view:
  win / draw / loss rates, goals for and against per match, touches per car per minute for each team."""
  P, N = env.ctx.carsPerTeam, env.numWorlds
  obs = env.reset()
  results, touches, ticks = [], torch.zeros(2), 0
  while len(results) < matches:
    obs, _, events = env.step(teamActions(blue(env.state, obs), orange(env.state, obs), P))
    touches += events["rewardTerms", "touch"].view(N, 2, P).sum((0, 2))
    ticks += env.actionRepeat
    for score in events["finalState", "score"][events["done"]].tolist():
      results.append(score)
  scores = torch.tensor(results[:matches], dtype=torch.float32)
  minutes = ticks * env.ctx.dt / 60
  return {
    "win": (scores[:, 0] > scores[:, 1]).float().mean().item(),
    "draw": (scores[:, 0] == scores[:, 1]).float().mean().item(),
    "loss": (scores[:, 0] < scores[:, 1]).float().mean().item(),
    "goalsFor": scores[:, 0].mean().item(), "goalsAgainst": scores[:, 1].mean().item(),
    "blueTouchesPerCarMin": touches[0].item() / (N * P * minutes),
    "orangeTouchesPerCarMin": touches[1].item() / (N * P * minutes),
  }
