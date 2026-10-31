from dataclasses import replace

import mujoco, pytest, numpy as np, torch

import carSoccer.config as cfg
from carSoccer.engine import quat
from carSoccer.engine.base import emptyGameState
from carSoccer.engine.sim.scene import ball, contacts, variants
from carSoccer.engine.sim.scene.builder import buildScene
from carSoccer.engine.sim.sensors import SensorModel
from carSoccer.engine.sim.simEngine import SimEngine
from carSoccer.env import makeEnv

@pytest.fixture(scope="module")
def scene():
  return buildScene(cfg.SceneConfig())

# ---------- car variants ----------

def testVariantsShareStructureAndDifferInParameters(scene):
  V = cfg.VariantConfig().count
  assert len(scene.variants) == len(scene.layout.geometry) == len(scene.cars) == V
  assert all(p.keys() == scene.variants[0].keys() for p in scene.variants)
  masses = {round(float(p["body_mass"][scene.layout.carBodies[0]].sum()), 4) for p in scene.variants}
  assert len(masses) == V                                    # every variant weighs something different
  assert len({g.wheelbase for g in scene.layout.geometry}) == V

def testVariantZeroIsNominal(scene):
  assert scene.cars[0] == cfg.CarConfig()
  np.testing.assert_array_equal(scene.variants[0]["geom_size"], scene.model.geom_size)

def testVariantWheelsStayUnderChassis():
  for car in variants.sampleCars(cfg.CarConfig(), replace(cfg.VariantConfig(), count=50, seed=3)):
    assert car.track / 2 + car.wheelRadius <= car.width / 2 and car.wheelbase / 2 + car.wheelRadius <= car.length / 2

def testStructureCheckRejectsDifferentModels(scene):
  with pytest.raises(ValueError):
    variants.checkSameStructure(scene.model, mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>"))

def testSetParamsAppliesVariantToOneWorld(scene):
  engine = SimEngine(scene, cfg.SceneConfig(), cfg.GameConfig(), 2, nthread=0)
  engine.setParams(torch.tensor([False, True]), {"variant": torch.tensor([0, 3])})
  models, chassis = engine.backend.models, scene.layout.carBodies[0][0]
  assert models[0] is scene.model
  assert models[1].body_mass[chassis] == pytest.approx(scene.variants[3]["body_mass"][chassis])
  assert engine.wheelbase[:, 0].tolist() == pytest.approx([scene.layout.geometry[0].wheelbase, scene.layout.geometry[3].wheelbase])
  engine.close()

# ---------- physics randomisation ----------

def testSolrefBatchMatchesScalar():
  e = torch.tensor([0.25, 0.4, 0.65, 0.8, 1.0])
  got = contacts.solrefBatch(e, 0.010, cfg.SimConfig())
  for k, value in enumerate(e.tolist()):
    np.testing.assert_allclose(got[k].numpy(), ball.contactSolref(value, cfg.SimConfig()), rtol=1e-4, atol=1e-9)

def testRandomPhysicsReachesTheModel():
  env = makeEnv(4, randomize=True, nthread=0, observe=False, seed=1)
  env.reset()
  p, L = env.state["params"], env.engine.scene.layout
  for w, model in enumerate(env.engine.backend.models):
    assert model.geom_friction[L.floorGeom, 0] == pytest.approx(p["floorFriction"][w].item(), rel=1e-6)
    assert model.body_mass[L.ballBody] == pytest.approx(p["ballMass"][w].item(), rel=1e-6)
    g = contacts.PAIRINGS.index("ballWall")
    expected = ball.contactSolref(p["restitution"][w, g].item(), cfg.SimConfig())
    np.testing.assert_allclose(model.pair_solref[L.pairIds["ballWall"][0]], expected, rtol=1e-4)
    drive, stall = int(L.driveAct[4, 0]), env.engine.scene.cars[p["variant"][w]].stallTorque
    assert model.actuator_gainprm[drive, 0] == pytest.approx(stall * p["motorStrength"][w, 4].item(), rel=1e-5)
  r = cfg.RandomizationConfig()
  assert ((p["latency"] >= r.latencyTicks[0]) & (p["latency"] <= r.latencyTicks[1])).all()
  env.close()

def testParamsRedrawnOnlyAtMatchStart():
  game = replace(cfg.GameConfig(), matchSeconds=6 / 50)       # two decisions per match
  env = makeEnv(2, game=game, randomize=True, nthread=0, observe=False)
  env.reset()
  first = env.state["params"].clone()
  env.step(torch.zeros(2, 6, 2))
  torch.testing.assert_close(env.state["params", "floorFriction"], first["floorFriction"])
  env.step(torch.zeros(2, 6, 2))                              # match ends: new draw
  assert not torch.equal(env.state["params", "floorFriction"], first["floorFriction"])
  env.close()

def testKickoffUsesVariantSpawnHeight():
  env = makeEnv(8, randomize=True, nthread=2, observe=False)
  env.reset()
  heights = env.ctx.spawnHeights[env.state["params", "variant"]]
  torch.testing.assert_close(env.state["car", "pos"][..., 2], heights[:, None].expand(8, 6), atol=1e-6, rtol=0)
  env.close()

# ---------- sensors ----------

def sensorState(N=2):
  state = emptyGameState(N, 3, torch.device("cpu"))
  state["car", "quat"][..., 0] = 1
  return state

def testPerfectSensorIsTheTruth(scene):
  engine = SimEngine(scene, cfg.SceneConfig(), cfg.GameConfig(), 1, nthread=0)
  engine.step(torch.ones(1, 6, 2))
  for key in ("car", "ball"):
    for leaf, value in engine.state[key].items():
      torch.testing.assert_close(engine.state["measured", key, leaf], value)
  engine.close()

def testSensorLagDelaysMeasurement():
  model = SensorModel(1, 3, cfg.SensorConfig(lagTicks=2), torch.Generator().manual_seed(0), torch.device("cpu"))
  state = sensorState(1)
  model.reset(torch.ones(1, dtype=torch.bool), state)
  seen = []
  for t in range(1, 5):
    state["ball", "pos"][0, 0] = t
    seen.append(model.measure(state)["ball", "pos"][0, 0].item())
  assert seen == [0, 0, 1, 2]

def testDropoutHoldsLastValue():
  sensors = cfg.SensorConfig(dropStart=1.0, dropEnd=0.0)    # drops on the first tick and never recovers
  model = SensorModel(1, 3, sensors, torch.Generator().manual_seed(0), torch.device("cpu"))
  state = sensorState(1)
  model.reset(torch.ones(1, dtype=torch.bool), state)
  state["car", "pos"][0, :, 0] = 5.0
  assert model.measure(state)["car", "pos"][0, :, 0].tolist() == [0.0] * 6

def testRealisticNoiseIsSmallAndBallNoisier():
  model = SensorModel(256, 3, replace(cfg.SensorConfig.realistic(), spikeProb=0, dropStart=0, lagTicks=0, lagJitterProb=0),
                      torch.Generator().manual_seed(0), torch.device("cpu"))
  state = sensorState(256)
  model.reset(torch.ones(256, dtype=torch.bool), state)
  m = model.measure(state)
  carStd, ballStd = m["car", "pos"].std().item(), m["ball", "pos"].std().item()
  assert 0.005 < carStd < 0.02 and ballStd > carStd
  assert quat.upZ(m["car", "quat"]).min() > 0.999            # heading noise is about z only
