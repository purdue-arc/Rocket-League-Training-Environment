import torch
from tensordict import TensorDict
from carSoccer.env import SoccerEnv
from carSoccer.learn.ppo import gae
from carSoccer.models.normalizer import RunningStd
from carSoccer.models.policy import AgentModel

class Collector:
  """Runs `steps` decisions across every world and returns a (T, B) rollout, B = N * 2P cars. Self-play: every car is
  the learner, with shared weights. Rewards are divided by a running std of the discounted return; a match that ends
  on the clock is bootstrapped with gamma * V(final observation)."""

  def __init__(self, env: SoccerEnv, model: AgentModel, steps: int, gamma: float, lam: float):
    self.env, self.model, self.steps, self.gamma, self.lam = env, model, steps, gamma, lam
    self.N, self.C = env.numWorlds, 2 * env.ctx.carsPerTeam
    self.B = self.N * self.C
    self.obs = env.reset()
    self.returnStd = RunningStd()
    self.discounted = torch.zeros(self.B)

  @torch.no_grad()
  def collect(self) -> tuple[TensorDict, dict[str, float]]:
    rows, stats = [], {"touch": 0.0, "goal": 0.0, "rewardRaw": 0.0, "ballSpeed": 0.0, "approach": 0.0, "matches": 0.0}
    for _ in range(self.steps):
      obs = self.obs
      self.model.updateNormalizers(obs)
      out = self.model(obs)
      action = out.dist.sample()
      self.obs, reward, events = self.env.step(action.view(self.N, self.C, 2))

      raw = reward.reshape(self.B)
      done = events["done"].repeat_interleave(self.C)
      truncated = events["truncated"].repeat_interleave(self.C)
      self.discounted = self.discounted * self.gamma * (~done).float() + raw
      self.returnStd.update(self.discounted)
      r = raw / self.returnStd.std
      if truncated.any():                  # outside the env step, so this sync is fine
        finalValue = self.model.value(events["finalObs"].reshape(self.B))
        r = r + self.gamma * finalValue * truncated.float()

      rows.append(TensorDict({
        "obs": obs, "action": action, "logProb": out.dist.log_prob(action).sum(-1), "value": out.value,
        "reward": r, "done": done,
      }, batch_size=[self.B]))
      self.accumulate(stats, events, raw)

    batch = torch.stack(rows)
    lastValue = self.model.value(self.obs)
    batch["advantage"], batch["return"] = gae(batch["reward"], batch["value"], batch["done"], lastValue, self.gamma, self.lam)
    return batch, self.summarise(stats)

  def accumulate(self, stats: dict, events: TensorDict, raw: torch.Tensor) -> None:
    state, terms = self.env.state, events["rewardTerms"]
    stats["touch"] += terms["touch"].sum().item()
    stats["goal"] += events["goal"].sum().item()
    stats["rewardRaw"] += raw.sum().item()
    stats["matches"] += events["done"].sum().item()
    stats["ballSpeed"] += state["ball", "vel"][:, :2].norm(dim=-1).mean().item()
    toBall = state["ball", "pos"][:, None, :2] - state["car", "pos"][..., :2]
    toBall = toBall / toBall.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    stats["approach"] += (state["car", "vel"][..., :2] * toBall).sum(-1).mean().item()

  def summarise(self, stats: dict) -> dict[str, float]:
    """Per-minute rates are per car (touches) or per world (goals), so they compare across world counts."""
    minutes = self.steps * self.env.actionRepeat * self.env.ctx.dt / 60
    return {
      "touchesPerCarMin": stats["touch"] / (self.B * minutes),
      "goalsPerWorldMin": stats["goal"] / (self.N * minutes),
      "rewardPerCarMin": stats["rewardRaw"] / (self.B * minutes),
      "ballSpeed": stats["ballSpeed"] / self.steps,
      "approachSpeed": stats["approach"] / self.steps,
      "matchesFinished": stats["matches"],
    }
