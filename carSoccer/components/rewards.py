import torch, carSoccer.config as cfg
from tensordict import TensorDict
from torch import Tensor
from carSoccer.components.ctx import Ctx
from carSoccer.engine.base import StepEvents

# pure terms: events of one decision -> (N, 2P) per car

def touchStart(touch: Tensor, prevTouch: Tensor) -> Tensor:
  """1 for a car that touches the ball this decision but did not last decision. Counting new touches, not ticks in
  contact, stops a car farming reward by pinning the ball against a wall."""
  return (touch & ~prevTouch).float()

def teamScored(goal: Tensor, team: Tensor) -> Tensor:
  """goal (N, 2) [blue scored, orange scored] -> 1 for every car whose team scored."""
  return goal[:, team].float()

def teamConceded(goal: Tensor, team: Tensor) -> Tensor:
  return goal[:, 1 - team].float()

class GameplayReward:
  """Per-car reward: each car is paid for starting a touch on the ball; every car of a team shares a large reward when
  the team scores and the same penalty when it concedes, so the goal term is zero-sum between teams."""

  def __init__(self, numWorlds: int, ctx: Ctx, weights: cfg.RewardConfig = cfg.RewardConfig()):
    P = ctx.carsPerTeam
    self.weights = weights
    self.team = torch.arange(2 * P, device=ctx.device) // P                  # (2P,) 0 blue, 1 orange
    self.prevTouch = torch.zeros(numWorlds, 2 * P, dtype=torch.bool, device=ctx.device)

  def reset(self, mask: Tensor) -> None:
    """Forget last decision's touches in worlds that kick off, so the first touch after a kickoff counts."""
    self.prevTouch = self.prevTouch & ~mask[:, None]

  def __call__(self, events: StepEvents) -> tuple[Tensor, TensorDict]:
    """Reward (N, 2P) for one decision, and the unweighted terms for logging."""
    w = self.weights
    terms = TensorDict({
      "touch": touchStart(events["ballTouch"], self.prevTouch),
      "goal": teamScored(events["goal"], self.team),
      "conceded": teamConceded(events["goal"], self.team),
    }, batch_size=events.batch_size)
    self.prevTouch = events["ballTouch"].clone()
    reward = w.touch * terms["touch"] + w.goal * terms["goal"] - w.conceded * terms["conceded"]
    return reward, terms
