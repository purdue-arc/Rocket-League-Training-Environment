import math

import pytest, torch

import carSoccer.config as cfg
from carSoccer.components.actions import ContinuousTeamActions
from carSoccer.components.bots import RandomDriver, chaseBall
from carSoccer.components.obs.frames import egoFrame, teamFrame, teamYaw
from carSoccer.env import makeEnv

P = cfg.SceneConfig().carsPerTeam

@pytest.fixture(scope="module")
def env():
  env = makeEnv(2, nthread=2)
  yield env
  env.close()

def leaves(td):
  return {k: v for k, v in td.items(include_nested=True, leaves_only=True)}

# ---------- frames ----------

def testEgoFrame():
  forward = egoFrame(torch.tensor([1.0, 0.0]), torch.tensor(math.pi / 2))   # car facing +y sees +x on its right
  assert forward.tolist() == pytest.approx([0.0, -1.0], abs=1e-6)

def testTeamFrameRotates180():
  sign = torch.tensor([1.0, -1.0])
  assert teamFrame(torch.tensor([[2.0, 1.0], [2.0, 1.0]]), sign).tolist() == [[2.0, 1.0], [-2.0, -1.0]]
  assert teamYaw(torch.tensor([0.0, 0.0]), sign).tolist() == pytest.approx([0.0, -math.pi])

# ---------- observations ----------

def testShapesMatchSpec(env):
  obs, spec, B = env.reset(), env.obsBuilder.spec, env.numWorlds * 2 * P
  assert obs.batch_size == torch.Size([B])
  assert obs["self"].shape == (B, spec.self)
  assert obs["mates"].shape == (B, P - 1, spec.car) and obs["opponents"].shape == (B, P, spec.car)
  assert obs["ball"].shape == (B, spec.ball) and obs["goals"].shape == (B, 2, spec.goal)
  assert obs["critic"].shape == (B, spec.critic)
  assert obs["slot"].tolist() == list(range(P)) * 2 * env.numWorlds

def testKickoffSymmetry(env):
  """Both teams see a mirrored kickoff identically in their own frame."""
  obs = env.reset().reshape(env.numWorlds, 2 * P)
  for key, value in leaves(obs).items():
    torch.testing.assert_close(value[:, :P], value[:, P:], atol=1e-5, rtol=0, msg=f"{key} differs between teams")

def testScriptedMatchObservationsFiniteAndInRange(env):
  """Scripted bots play a full 60 s match; it ends on the last decision and every observation stays finite and bounded."""
  env.reset()
  decisions = env.ctx.matchTicks // env.actionRepeat
  for k in range(decisions):
    obs, _, events = env.step(chaseBall(env.state, env.ctx))
    assert events["done"].tolist() == [k == decisions - 1] * env.numWorlds
    for key, value in leaves(obs).items():
      if value.dtype.is_floating_point:
        assert torch.isfinite(value).all(), key
        assert value.abs().max() < 50, key

def testFinalObsIsPerWorld(env):
  env.reset()
  _, _, events = env.step(torch.zeros(env.numWorlds, 2 * P, 2))
  assert events["finalObs"].batch_size == torch.Size([env.numWorlds, 2 * P])

# ---------- actions ----------

def testTeamActionsReshapeToCars(env):
  raw = torch.arange(env.numWorlds * 2 * 2 * P, dtype=torch.float32).view(env.numWorlds, 2, 2 * P) / 100
  cars = ContinuousTeamActions(P).parse(raw, env.ctx)
  assert cars.shape == (env.numWorlds, 2 * P, 2)
  assert cars[0, P].tolist() == pytest.approx(raw[0, 1, :2].tolist())    # orange0 = orange team's first pair

def testRandomDriverMovesEveryCar():
  env = makeEnv(1, nthread=0, observe=False, seed=3)
  env.reset()
  start = env.state["car", "pos"].clone()
  driver = RandomDriver(1, P, env.ctx)
  for _ in range(50):                                   # 3 s
    env.step(driver(env.state))
  moved = (env.state["car", "pos"] - start)[..., :2].norm(dim=-1)
  assert (moved > 0.3).all(), moved
  env.close()
