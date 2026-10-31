import torch, carSoccer.config as cfg
from tensordict import TensorDict
from torch import Tensor
from carSoccer.components.ctx import Ctx
from carSoccer.engine.base import GameState, StepEvents

# pure terms: events (or state) of one decision -> (N, 2P) per car

def touchStart(touch: Tensor, prevTouch: Tensor) -> Tensor:
  """1 for a car that touches the ball this decision but did not last decision. Counting new touches, not ticks in
  contact, stops a car farming reward by pinning the ball against a wall."""
  return (touch & ~prevTouch).float()

def splitTouchByEffect(touch: Tensor, ballSpeed: Tensor, stallSpeed: float) -> tuple[Tensor, Tensor]:
  """touch (N, 2P), freshly computed by touchStart, split into (useful, useless) by whether the ball's speed right
  after contact clears stallSpeed. touchStart alone only proves contact happened; this is what tells a real hit
  from a graze, or from a touch that chatters on and off while the ball is pinned uselessly in a corner with
  nowhere to go - either way the ball's speed stays near zero despite the "fresh touch" firing again and again."""
  moving = (ballSpeed > stallSpeed)[:, None]
  return touch * moving.float(), touch * (~moving).float()

def teamScored(goal: Tensor, team: Tensor) -> Tensor:
  """goal (N, 2) [blue scored, orange scored] -> 1 for every car whose team scored."""
  return goal[:, team].float()

def teamConceded(goal: Tensor, team: Tensor) -> Tensor:
  return goal[:, 1 - team].float()

def approachSpeed(carPos: Tensor, carVel: Tensor, ballPos: Tensor) -> Tensor:
  """Component of each car's own velocity toward the ball right now, m/s (N, 2P). Dense version of "touch": rewards
  closing the gap even before a touch happens, so a from-scratch policy gets gradient before it ever bumps the ball."""
  toBall = ballPos[:, None, :2] - carPos[..., :2]
  toBall = toBall / toBall.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  return (carVel[..., :2] * toBall).sum(-1)

def ballToGoalSpeed(ballPos: Tensor, ballVel: Tensor, team: Tensor, arena: cfg.ArenaConfig) -> Tensor:
  """Component of the ball's velocity toward each car's attacking goal, m/s (N, 2P): same value for every car of a
  team (zero-sum between teams, like goal/conceded), rewarding actually driving the ball goalward, not just touching it."""
  sign = torch.where(team.bool(), -1.0, 1.0)
  goalXY = torch.stack([sign * arena.length / 2, torch.zeros_like(sign)], -1)
  toGoal = goalXY[None] - ballPos[:, None, :2]
  toGoal = toGoal / toGoal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  return (ballVel[:, None, :2] * toGoal).sum(-1)

class GameplayReward:
  """Per-car reward: each car is paid for starting a touch on the ball and for closing speed on it; every car of a
  team shares a large reward when the team scores (and the same penalty when it concedes) plus a small one for
  driving the ball toward the opponent's goal, so both goal terms are zero-sum between teams."""

  def __init__(self, numWorlds: int, ctx: Ctx, weights: cfg.RewardConfig = cfg.RewardConfig()):
    P = ctx.carsPerTeam
    self.weights, self.arena = weights, ctx.arena
    self.team = torch.arange(2 * P, device=ctx.device) // P 
    self.prevTouch = torch.zeros(numWorlds, 2 * P, dtype=torch.bool, device=ctx.device)

  def reset(self, mask: Tensor) -> None:
    """Forget last decision's touches in worlds that kick off, so the first touch after a kickoff counts."""
    self.prevTouch = self.prevTouch & ~mask[:, None]

  def __call__(self, events: StepEvents) -> tuple[Tensor, TensorDict]:
    """Reward (N, 2P) for one decision, and the unweighted terms for logging. Reads the true post-step state
    (positions/velocities) from events["state"] if the caller put one there (env.py does); the dense shaping terms
    and the useful/useless touch split are all zero/unsplit without it, e.g. in tests that only exercise the sparse
    touch/goal terms and in any reward that doesn't set it."""
    w = self.weights
    touch = touchStart(events["ballTouch"], self.prevTouch)
    state: GameState | None = events.get("state", None)
    if state is not None:
      ballSpeed = state["ball", "vel"][..., :2].norm(dim=-1)
      usefulTouch, uselessTouch = splitTouchByEffect(touch, ballSpeed, w.stallSpeed)
    else:
      usefulTouch, uselessTouch = touch, torch.zeros_like(touch)
    terms = {
      "touch": usefulTouch,
      "stuckTouch": uselessTouch,
      "goal": teamScored(events["goal"], self.team),
      "conceded": teamConceded(events["goal"], self.team),
    }
    if state is not None:
      terms["approach"] = approachSpeed(state["car", "pos"], state["car", "vel"], state["ball", "pos"])
      terms["ballToGoal"] = ballToGoalSpeed(state["ball", "pos"], state["ball", "vel"], self.team, self.arena)
    else:
      terms["approach"] = terms["ballToGoal"] = torch.zeros_like(touch)
    terms = TensorDict(terms, batch_size=events.batch_size)
    self.prevTouch = events["ballTouch"].clone()
    reward = (w.touch * terms["touch"] - w.stuckPenalty * terms["stuckTouch"]
              + w.goal * terms["goal"] - w.conceded * terms["conceded"]
              + w.approach * terms["approach"] + w.ballToGoal * terms["ballToGoal"])
    return reward, terms
