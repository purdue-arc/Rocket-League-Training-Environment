import torch
from tensordict import TensorDict
from carSoccer.env import SoccerEnv
from carSoccer.learn.ppo import gae
from carSoccer.learn.pool import OpponentPool
from carSoccer.models.normalizer import RunningStd
from carSoccer.models.policy import AgentModel

class Collector:
  """Runs `steps` decisions across every world and returns a (T, B) rollout, B = N * 2P cars. Self-play: every car is
  the learner, with shared weights, except for worlds this iteration hands to a frozen OpponentPool snapshot (see
  chooseOpponent): those cars' rows are marked "trainable" = False and PPO drops them, since their actions didn't
  come from the live policy's distribution. Rewards are divided by a running std of the discounted return; a match
  that ends on the clock is bootstrapped with gamma * V(final observation)."""

  def __init__(self, env: SoccerEnv, model: AgentModel, steps: int, gamma: float, lam: float,
               pool: OpponentPool | None = None, poolProb: float = 0.0, poolFrac: float = 0.5,
               noiseRange: tuple[float, float] = (0.85, 1.15)):
    self.env, self.model, self.steps, self.gamma, self.lam = env, model, steps, gamma, lam
    self.pool, self.poolProb, self.poolFrac, self.noiseRange = pool, poolProb, poolFrac, noiseRange
    self.N, self.P, self.C = env.numWorlds, env.ctx.carsPerTeam, 2 * env.ctx.carsPerTeam
    self.B = self.N * self.C
    self.team = torch.arange(self.C, device=env.device) // self.P
    self.obs = env.reset()
    self.returnStd = RunningStd()
    self.discounted = torch.zeros(self.B)

  def chooseOpponent(self) -> tuple[object | None, torch.Tensor]:
    """With probability poolProb, sample one frozen snapshot and hand it a random poolFrac slice of worlds, each
    giving it a random side (blue or orange) so the pool doesn't systematically weaken one side. Its action std is
    scaled by a factor drawn uniformly from noiseRange, centred so the long-run average opponent is exactly as
    strong as it was at checkpoint time - some matches it plays sharper, some noisier, never biased either way.
    Returns (opponent or None, car mask (N, C) of cars it controls this iteration)."""
    carMask = torch.zeros(self.N, self.C, dtype=torch.bool, device=self.team.device)
    if self.pool is None or len(self.pool) == 0 or torch.rand(()).item() >= self.poolProb:
      return None, carMask
    opponent = self.pool.sample(self.model)
    poolWorlds = torch.rand(self.N) < self.poolFrac
    frozenSide = torch.randint(0, 2, (self.N,))                    # which team (0 blue, 1 orange) this world freezes
    carMask = poolWorlds[:, None] & (self.team[None, :] == frozenSide[:, None])
    return opponent, carMask

  @torch.no_grad()
  def collect(self) -> tuple[TensorDict, dict[str, float]]:
    opponent, oppMask = self.chooseOpponent()
    trainable = (~oppMask).reshape(self.B)
    stdScale = torch.empty(()).uniform_(*self.noiseRange).item() if opponent is not None else 1.0

    rows, stats = [], {"touch": 0.0, "goal": 0.0, "rewardRaw": 0.0, "ballSpeed": 0.0, "approach": 0.0, "matches": 0.0}
    for _ in range(self.steps):
      obs = self.obs
      self.model.updateNormalizers(obs)
      out = self.model(obs)
      action = out.dist.sample()
      carActions = action.view(self.N, self.C, 2)
      if opponent is not None:
        oppAction = opponent.act(obs, deterministic=False, stdScale=stdScale).view(self.N, self.C, 2)
        carActions = torch.where(oppMask[..., None], oppAction, carActions)
      self.obs, reward, events = self.env.step(carActions)

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
        "reward": r, "done": done, "trainable": trainable,
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
