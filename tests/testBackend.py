from dataclasses import replace

import mujoco, pytest, numpy as np, torch

import carSoccer.config as cfg
from carSoccer.engine.sim.backend.cpuRollout import CpuRolloutBackend
from carSoccer.engine.sim.scene import ball
from carSoccer.engine.sim.scene.builder import buildScene

GRAVITY = 9.81                           # m/s^2
E_TOL = 0.04                             # same tolerance as the M0 ball tests

@pytest.fixture(scope="module")
def scene():
  return buildScene(cfg.SceneConfig())

def ballDropBackend(numWorlds, ballCfg=None, dropHeight=1.0):
  model = mujoco.MjModel.from_xml_string(ball.ballDropXML(ballCfg or cfg.BallConfig(), cfg.SimConfig(), dropHeight))
  return model, CpuRolloutBackend(model, numWorlds, nthread=2)

def throwDown(backend, speeds):
  """Ball 1 mm above the floor moving down at speeds[w] in world w; returns the peak rebound speed / speed per world."""
  radius = cfg.BallConfig().radius
  qpos, qvel = backend.qpos.clone(), backend.qvel.clone()
  qpos[:, 2] = radius + 0.001
  qvel[:, 2] = -torch.tensor(speeds, dtype=torch.float64)
  backend.writeState(torch.ones(backend.numWorlds, dtype=torch.bool), qpos, qvel)
  peak = torch.zeros(backend.numWorlds, dtype=torch.float64)
  for _ in range(40):                     # 40 ms covers the 10 ms contact
    backend.step(torch.zeros(backend.numWorlds, 0, dtype=torch.float64), None, 1)
    peak = torch.maximum(peak, backend.qvel[:, 2])
  return (peak / torch.tensor(speeds, dtype=torch.float64)).numpy()

# ---------- parity ----------

def testParityWithMjStep(scene):
  model, N, nsub = scene.model, 4, 20
  rng = np.random.default_rng(0)
  ctrl = rng.uniform(-1, 1, (N, model.nu)) * np.where(np.arange(model.nu) % 6 < 2, 0.4, 1.0)   # steering within range
  backend = CpuRolloutBackend(model, N, nthread=2)
  for _ in range(50):                    # 1 s of game time
    backend.step(torch.from_numpy(ctrl), None, nsub)
  for w in range(N):
    data = mujoco.MjData(model)
    data.ctrl[:] = ctrl[w]
    for _ in range(50 * nsub):
      mujoco.mj_step(model, data)
    np.testing.assert_allclose(backend.qpos[w].numpy(), data.qpos, atol=3e-3)
  backend.close()

def testStepUpdatesViewsInPlace(scene):
  backend = CpuRolloutBackend(scene.model, 2, nthread=1)
  view = backend.qpos
  ballZ = scene.layout.ballQpos + 2
  qpos = backend.qpos.clone()
  qpos[:, ballZ] = 1.0
  backend.writeState(torch.ones(2, dtype=torch.bool), qpos, backend.qvel.clone())
  backend.step(torch.zeros(2, scene.model.nu, dtype=torch.float64), None, 20)
  assert view is backend.qpos and view[0, ballZ] < 1.0
  backend.close()

# ---------- the M0 ball tests through the backend ----------

def testFreeFallThroughBackend():
  model, backend = ballDropBackend(2, dropHeight=2.0)
  for _ in range(15):
    backend.step(torch.zeros(2, 0, dtype=torch.float64), None, 20)
  t = 0.3
  np.testing.assert_allclose(backend.qvel[:, 2].numpy(), -GRAVITY * t, rtol=1e-4)
  np.testing.assert_allclose(backend.qpos[:, 2].numpy(), 2.0 - 0.5 * GRAVITY * t ** 2, atol=5e-3)
  backend.close()

def testBallComesToRestThroughBackend():
  model, backend = ballDropBackend(1, dropHeight=0.5)
  for _ in range(250):
    backend.step(torch.zeros(1, 0, dtype=torch.float64), None, 20)
  assert backend.qvel[0, :3].norm() < 1e-2
  assert backend.qpos[0, 2] == pytest.approx(cfg.BallConfig().radius, abs=5e-4)
  backend.close()

def testRestitutionThroughBackend():
  model, backend = ballDropBackend(3)
  e = throwDown(backend, [3.0, 8.0, 15.0])
  np.testing.assert_allclose(e, cfg.BallConfig().restitutionFloor, atol=E_TOL)
  backend.close()

# ---------- masks and per-world parameters ----------

def testWriteStateOnlyTouchesMaskedWorlds():
  model, backend = ballDropBackend(3)
  qpos = backend.qpos.clone()
  qpos[:, 2] = 0.7
  backend.writeState(torch.tensor([False, True, False]), qpos, backend.qvel.clone())
  assert backend.qpos[:, 2].tolist() == [1.0, 0.7, 1.0]
  backend.close()

def testSetParamsPerWorld():
  model, backend = ballDropBackend(2)
  g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
  solref = torch.from_numpy(np.tile(model.geom_solref, (2, 1, 1)))
  solref[1, g] = torch.tensor(ball.contactSolref(0.4, cfg.SimConfig()))
  backend.setParams(torch.tensor([False, True]), {"geom_solref": solref})
  e = throwDown(backend, [8.0, 8.0])
  assert e[0] == pytest.approx(0.65, abs=E_TOL) and e[1] == pytest.approx(0.4, abs=E_TOL)
  assert backend.models[0] is model and backend.models[1] is not model
  backend.close()

def testSensorPeakCatchesContactWithinSubsteps(scene):
  backend = CpuRolloutBackend(scene.model, 1, nthread=1)
  L, c, r = scene.layout, cfg.CarConfig(), cfg.BallConfig().radius
  qpos, qvel = backend.qpos.clone(), backend.qvel.clone()
  qpos[0, L.ballQpos:L.ballQpos + 3] = torch.tensor([-2, -1.5 - c.width / 2 - r - 0.001, r])
  qvel[0, L.ballQvel + 1] = 3.0
  backend.writeState(torch.ones(1, dtype=torch.bool), qpos, qvel)
  backend.step(torch.zeros(1, scene.model.nu, dtype=torch.float64), None, 20)
  assert backend.sensordata[0, L.touchSensor].tolist() == [1, 0, 0, 0, 0, 0]
  backend.close()
