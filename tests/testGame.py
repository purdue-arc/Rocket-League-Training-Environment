import math, time
from dataclasses import replace

import pytest, torch

import carSoccer.config as cfg
from carSoccer.engine import quat
from carSoccer.engine.sim import physics
from carSoccer.engine.sim.scene import car
from carSoccer.engine.sim.scene.builder import buildScene
from carSoccer.engine.sim.simEngine import SimEngine
from carSoccer.env import makeEnv

@pytest.fixture(scope="module")
def scene():
  return buildScene(cfg.SceneConfig())

def makeEngine(scene, numWorlds=1, game=cfg.GameConfig(), backend=None):
  return SimEngine(scene, cfg.SceneConfig(), game, numWorlds, nthread=0, backend=backend)

def idle(engine):
  return torch.zeros(engine.numWorlds, 2 * engine.P, 2)

def teleportBall(engine, pos, vel=(0, 0, 0)):
  target = engine.state.clone()
  target["ball", "pos"] = torch.tensor([pos], dtype=torch.float32).expand(engine.numWorlds, 3)
  target["ball", "vel"] = torch.tensor([vel], dtype=torch.float32).expand(engine.numWorlds, 3)
  engine.setState(torch.ones(engine.numWorlds, dtype=torch.bool), target)

# ---------- physics helpers ----------

def testAckermannSplitMatchesScalar():
  g = cfg.CarConfig()
  left, right = physics.ackermannSplit(torch.tensor([[0.3, -0.2]]), torch.tensor([[g.wheelbase]]), torch.tensor([[g.track]]))
  for k, steer in enumerate((0.3, -0.2)):
    assert (left[0, k].item(), right[0, k].item()) == pytest.approx(car.ackermannAngles(steer, g.wheelbase, g.track), abs=1e-6)

def testBallAeroDirections():
  game, ball = cfg.GameConfig(), cfg.BallConfig()
  f = physics.ballAero(torch.tensor([[5.0, 0, 0]]), torch.tensor([[0, 0, 10.0]]), ball, game)[0]
  assert f[0] < 0                                       # drag opposes motion
  assert f[1] > 0                                       # spin about +z curls a +x ball toward +y
  expectedDrag = 0.5 * game.airDensity * game.ballDrag * math.pi * ball.radius ** 2 * 25
  assert -f[0].item() == pytest.approx(expectedDrag, rel=1e-5)

def testMotorDeadbandAndLag():
  game = cfg.GameConfig()
  zero = torch.zeros(1)
  assert physics.motorStep(torch.tensor([game.throttleDeadband / 2]), zero, game, 0.02).item() == 0
  u = physics.motorStep(torch.ones(1), zero, game, 0.02).item()
  assert u == pytest.approx(1 - math.exp(-0.02 / game.motorTau))

# ---------- rules ----------

def testGoalFiresExactlyOnce(scene):
  engine = makeEngine(scene)
  X, r = cfg.ArenaConfig().length / 2, cfg.BallConfig().radius
  teleportBall(engine, (X - 0.3, 0, r), vel=(3, 0, 0))
  goals = torch.zeros(2, dtype=torch.long)
  for _ in range(100):                                  # ball goes in, bounces off the back wall, sits in the goal
    goals += engine.step(idle(engine))["goal"][0].long()
  assert goals.tolist() == [1, 0]
  assert engine.state["score"][0].tolist() == [1, 0]
  engine.close()

def testGoalKicksOffAndLatchClears():
  env = makeEnv(1, nthread=0, observe=False)
  env.reset()
  X, r = cfg.ArenaConfig().length / 2, cfg.BallConfig().radius
  teleportBall(env.engine, (-X - r - 0.05, 0, r))       # inside blue's goal: orange scores
  _, _, events = env.step(torch.zeros(1, 6, 2))
  assert events["goal"][0].tolist() == [False, True] and events["kickoff"][0]
  assert env.state["score"][0].tolist() == [0, 1]
  assert env.state["ball", "pos"][0, :2].abs().max() < 1e-3          # back on the centre spot
  assert not env.engine.goalLatch[0]
  env.close()

def testBallOutKicksOff():
  env = makeEnv(1, nthread=0, observe=False)
  env.reset()
  teleportBall(env.engine, (0, cfg.ArenaConfig().width / 2 + 2, 1.0))
  _, _, events = env.step(torch.zeros(1, 6, 2))
  assert events["ballOut"][0] and events["kickoff"][0]
  assert env.state["ball", "pos"][0, :2].abs().max() < 1e-3
  env.close()

def testFlipRecovery(scene):
  engine = makeEngine(scene)
  target = engine.state.clone()
  target["car", "quat"][0, 0] = torch.tensor([0.0, 1, 0, 0])          # blue0 on its roof
  target["car", "pos"][0, 0, 2] = 0.2
  engine.setState(torch.ones(1, dtype=torch.bool), target)
  flippedAt = None
  for tick in range(80):
    events = engine.step(idle(engine))
    if flippedAt is None and events["flipped"][0, 0]:
      flippedAt = tick
  assert flippedAt == 0
  assert quat.upZ(engine.state["car", "quat"][0, 0]) > 0.99
  assert engine.state["car", "uprightTime"][0, 0] == 0
  engine.close()

def testLatencyDelaysActions(scene):
  engine = makeEngine(scene, numWorlds=2)
  engine.setParams(torch.tensor([False, True]), {"latency": torch.tensor([0, 2])})
  action = idle(engine)
  action[..., 0] = 1.0
  moved = []
  for _ in range(3):
    engine.step(action)
    moved.append((engine.throttle[:, 0] > 0).tolist())
  assert moved == [[True, False], [True, False], [True, True]]
  engine.close()

# ---------- env ----------

def testMidRepeatFreeze():
  game = replace(cfg.GameConfig(), matchSeconds=4 / 50)  # match lasts 4 ticks; a decision is 3
  env = makeEnv(1, game=game, nthread=0, observe=False)
  env.reset()
  _, _, events = env.step(torch.ones(1, 6, 2))
  assert not events["done"][0]
  _, _, events = env.step(torch.ones(1, 6, 2))
  assert events["done"][0] and events["truncated"][0]
  assert events["finalState", "clock"][0] == 4          # captured on the final whistle, not at the end of the decision
  assert env.state["clock"][0] == 0                     # auto-reset
  assert env.engine.throttle.abs().max() == 0           # and the new match starts from rest
  env.close()

class NoPhysicsBackend:
  """Stands in for the backend so the timing test sees only the engine's own tensor work."""

  def __init__(self, model, numWorlds):
    self.numWorlds, self.device = numWorlds, torch.device("cpu")
    self.qvel = torch.zeros(numWorlds, model.nv, dtype=torch.float64)
    self.qpos0 = torch.from_numpy(model.qpos0.copy())
    self.qpos = self.qpos0.expand(numWorlds, -1).clone()
    self.sensordata = torch.zeros(numWorlds, model.nsensordata, dtype=torch.float64)

  def step(self, ctrl, xfrc, nsub): pass
  def writeState(self, mask, qpos, qvel):
    self.qpos.copy_(torch.where(mask[:, None], qpos, self.qpos))
  def setParams(self, mask, params): pass
  def close(self): pass

def testNoPerWorldLoopInStep(scene):
  def perWorld(N):
    engine = makeEngine(scene, N, backend=NoPhysicsBackend(scene.model, N))
    action = idle(engine)
    engine.step(action)
    start = time.perf_counter()
    for _ in range(20):
      engine.step(action)
    return (time.perf_counter() - start) / 20 / N
  assert perWorld(1024) < perWorld(64) / 2            # a per-world Python loop would keep this flat
