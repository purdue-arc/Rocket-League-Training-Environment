from dataclasses import replace

import pytest, torch
from tensordict import TensorDict

import carSoccer.config as cfg
from carSoccer.env import makeEnv
from carSoccer.learn.collector import Collector
from carSoccer.learn.play import evaluate, makeController
from carSoccer.learn.ppo import PPO, PPOConfig, gae
from carSoccer.models.encoders import flattenObs
from carSoccer.models.normalizer import RunningNorm, RunningStd
from carSoccer.models.policy import AgentModel, ModelSpec

P = cfg.SceneConfig().carsPerTeam

@pytest.fixture(scope="module")
def env():
  env = makeEnv(2, nthread=0)
  yield env
  env.close()

# ---------- pieces ----------

def testGaeByHand():
  rewards = torch.tensor([[1.0], [0.0], [2.0]])
  values = torch.tensor([[0.5], [0.5], [0.5]])
  dones = torch.tensor([[False], [True], [False]])
  adv, ret = gae(rewards, values, dones, torch.tensor([1.0]), gamma=0.9, lam=1.0)
  # episode 1 ends at t=1: returns 1 + 0.9 * 0 = 1.0 and 0; episode 2 bootstraps from lastValue: 2 + 0.9 * 1
  assert ret[:, 0].tolist() == pytest.approx([1.0, 0.0, 2.9])
  assert adv[:, 0].tolist() == pytest.approx([0.5, -0.5, 2.4])

def testRunningNormMatchesBatchStats():
  norm, data = RunningNorm(3), torch.randn(1000, 3) * torch.tensor([1.0, 5.0, 0.1]) + 2
  for chunk in data.split(100):
    norm.update(chunk)
  torch.testing.assert_close(norm.mean, data.mean(0), atol=1e-4, rtol=0)
  torch.testing.assert_close(norm.var, data.var(0, unbiased=False), atol=1e-3, rtol=1e-4)
  assert norm(data).mean(0).abs().max() < 1e-3

def testRunningStd():
  s = RunningStd()
  s.update(torch.randn(10000) * 3)
  assert s.std == pytest.approx(3, rel=0.05)

def testModelShapesAndThrottlePrior(env):
  obs = env.reset()
  model = AgentModel(env.obsBuilder.spec, P)
  out = model(obs)
  B = env.numWorlds * 2 * P
  assert out.dist.mean.shape == (B, 2) and out.value.shape == (B,)
  assert out.dist.mean[:, 0].mean().item() == pytest.approx(ModelSpec().throttleBias, abs=0.05)
  assert out.dist.stddev[0].detach().tolist() == pytest.approx([ModelSpec().initStd] * 2)

def testExportActorRoundTrip(env, tmp_path):
  obs = env.reset()
  model = AgentModel(env.obsBuilder.spec, P)
  model.updateNormalizers(obs)
  path = str(tmp_path / "actor.pt")
  model.exportActor(path)
  actor = torch.jit.load(path)
  torch.testing.assert_close(actor(flattenObs(obs, P)), model.act(obs).clamp(-1, 1))

# ---------- PPO end to end ----------

class ThrottleReward:
  """Trivial task (guide Section 3): reward the effective motor duty, so the policy only has to learn full throttle."""

  def __init__(self, env):
    self.env = env

  def reset(self, mask):
    pass

  def __call__(self, events):
    reward = self.env.engine.throttle.clone()
    return reward, TensorDict({"touch": torch.zeros_like(reward)}, batch_size=events.batch_size)

def testPpoLearnsFullThrottle():
  torch.manual_seed(0)
  env = makeEnv(8, game=replace(cfg.GameConfig(), matchSeconds=5.0), scatterProb=1.0, nthread=4)
  env.reward = ThrottleReward(env)
  model = AgentModel(env.obsBuilder.spec, P, ModelSpec(hidden=(64, 64), criticHidden=(64, 64)))
  ppo = PPO(model, PPOConfig(lr=1e-3, minibatches=4))
  collector = Collector(env, model, steps=16, gamma=0.99, lam=0.95)
  before = model.act(collector.obs)[:, 0].mean().item()
  for _ in range(20):
    batch, _ = collector.collect()
    ppo.update(batch)
  after = model.act(collector.obs)[:, 0].mean().item()
  env.close()
  assert before < 0.4 and after > 0.9

def testCollectorBatchShapes(env):
  model = AgentModel(env.obsBuilder.spec, P)
  batch, stats = Collector(env, model, steps=4, gamma=0.99, lam=0.95).collect()
  B = env.numWorlds * 2 * P
  assert batch.batch_size == torch.Size([4, B])
  assert batch["action"].shape == (4, B, 2) and batch["advantage"].shape == (4, B)
  assert torch.isfinite(batch["return"]).all()
  assert set(stats) >= {"touchesPerCarMin", "goalsPerWorldMin", "approachSpeed"}

def testEvaluateRatesSumToOne():
  env = makeEnv(4, game=replace(cfg.GameConfig(), matchSeconds=1.2), nthread=2)
  result = evaluate(env, makeController("chase", env), makeController("idle", env), matches=4)
  env.close()
  assert result["win"] + result["draw"] + result["loss"] == pytest.approx(1)
  assert result["orangeTouchesPerCarMin"] <= result["blueTouchesPerCarMin"]
