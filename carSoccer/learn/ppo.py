import torch
from dataclasses import dataclass
from tensordict import TensorDict
from torch import Tensor, nn
from carSoccer.models.policy import AgentModel

@dataclass(frozen=True)
class PPOConfig:
  lr: float = 3e-4
  gamma: float = 0.99                      # per decision (about 6 s at 17 Hz)
  lam: float = 0.95
  clip: float = 0.2
  valueClip: float = 0.2
  epochs: int = 4
  minibatches: int = 8
  entropyCoef: float = 0.003
  valueCoef: float = 0.5
  maxGradNorm: float = 0.5
  targetKl: float = 0.03                   # stop the epoch loop early once the policy moves this far

def gae(rewards: Tensor, values: Tensor, dones: Tensor, lastValue: Tensor, gamma: float, lam: float) -> tuple[Tensor, Tensor]:
  """Generalised advantage estimation over (T, B). dones[t] means decision t ended its episode, so nothing is
  bootstrapped from the next observation (a truncated episode already has gamma * V(final) in its reward)."""
  T = rewards.shape[0]
  advantages = torch.zeros_like(rewards)
  last = torch.zeros_like(lastValue)
  for t in reversed(range(T)):
    nextValue = lastValue if t == T - 1 else values[t + 1]
    alive = 1.0 - dones[t].float()
    delta = rewards[t] + gamma * nextValue * alive - values[t]
    last = delta + gamma * lam * alive * last
    advantages[t] = last
  return advantages, advantages + values

class PPO:
  def __init__(self, model: AgentModel, cfg: PPOConfig = PPOConfig()):
    self.model, self.cfg = model, cfg
    self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

  def update(self, batch: TensorDict) -> dict[str, float]:
    """One learning step on a full rollout (T, B): obs, action, logProb, value, advantage, return. Rows with
    trainable = False (an OpponentPool snapshot's cars, if any: see Collector.chooseOpponent) are dropped - their
    actions weren't drawn from the live policy, so they'd corrupt the PPO ratio rather than train it."""
    cfg, flat = self.cfg, batch.reshape(-1)
    flat = flat[flat["trainable"]]
    n = flat.batch_size[0]
    size = n // cfg.minibatches
    stats = {"policyLoss": 0.0, "valueLoss": 0.0, "entropy": 0.0, "kl": 0.0, "clipFrac": 0.0}
    updates = 0
    for epoch in range(cfg.epochs):
      perm = torch.randperm(n)
      kls = []
      for start in range(0, size * cfg.minibatches, size):
        mb = flat[perm[start:start + size]]
        out = self.model(mb["obs"])
        logProb = out.dist.log_prob(mb["action"]).sum(-1)
        ratio = (logProb - mb["logProb"]).exp()
        adv = mb["advantage"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policyLoss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        clipped = mb["value"] + (out.value - mb["value"]).clamp(-cfg.valueClip, cfg.valueClip)
        valueLoss = 0.5 * torch.max((out.value - mb["return"]) ** 2, (clipped - mb["return"]) ** 2).mean()
        entropy = out.dist.entropy().sum(-1).mean()
        loss = policyLoss + cfg.valueCoef * valueLoss - cfg.entropyCoef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.maxGradNorm)
        self.optimizer.step()

        with torch.no_grad():
          kl = ((ratio - 1) - (logProb - mb["logProb"])).mean().item()
          kls.append(kl)
          for key, value in (("policyLoss", policyLoss), ("valueLoss", valueLoss), ("entropy", entropy)):
            stats[key] += value.item()
          stats["kl"] += kl
          stats["clipFrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
        updates += 1
      if sum(kls) / len(kls) > cfg.targetKl:
        break

    stats = {k: v / updates for k, v in stats.items()}
    stats["epochs"] = epoch + 1
    ret, value = flat["return"], flat["value"]
    stats["explainedVariance"] = (1 - (ret - value).var() / (ret.var() + 1e-8)).item()
    stats["std"] = self.model.logStd.exp().mean().item()
    return stats
