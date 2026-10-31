from dataclasses import replace

import mujoco
import numpy as np
import pytest

import carSoccer.config as cfg

GRAVITY = 9.81                           # m/s^2
E_TOL = 0.04                             # allowed restitution error with >= 10 steps per contact (sweep experiment E)
E_TOL_FINE = 0.01                        # allowed error when the contact is resolved in ~100 steps
REST_TOL = 0.0005                        # m; resting ball sinks well under 0.1 mm (sweep experiment E)

def measureRestitution(model, vIn, maxTime=1.0):
  """Throw the ball down at vIn (m/s) from just above the floor; return bounce speed / vIn."""
  data = mujoco.MjData(model)
  g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
  radius = model.geom_size[g][0]
  
  data.qpos[2] = radius + 0.001      # 1 mm gap: gravity barely changes vIn before impact
  data.qvel[2] = -vIn
  
  bounced, peak = False, 0.0
  for _ in range(round(maxTime / model.opt.timestep)):
    mujoco.mj_step(model, data)
    vz = data.qvel[2]
    if vz > 0:
      bounced = True
      if vz > peak:
        peak = vz
      elif data.ncon == 0:       # left the floor and slowing down: peak passed
        break
    elif bounced:
      break
  assert bounced, f"ball never bounced within {maxTime} s at vIn={vIn}"
  return peak / vIn

def testGravity(model):
  np.testing.assert_allclose(model.opt.gravity, [0.0, 0.0, -GRAVITY], rtol=1e-6)

def testFreeFall(makeModel):
  model = makeModel(dropHeight=2.0)
  data = mujoco.MjData(model)
  for _ in range(round(0.3 / model.opt.timestep)):     # lands after ~0.6 s, so still airborne
    mujoco.mj_step(model, data)
  t = data.time
  assert data.qvel[2] == pytest.approx(-GRAVITY * t, rel=1e-4)
  # position drifts ~g*t*dt/2 from the exact parabola due to time stepping
  assert data.qpos[2] == pytest.approx(2.0 - 0.5 * GRAVITY * t**2, abs=5e-3)

def testBallComesToRest(makeModel):
  model = makeModel(dropHeight=0.5)
  data = mujoco.MjData(model)
  for _ in range(round(5.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
  assert np.linalg.norm(data.qvel[:3]) < 1e-2
  assert data.qpos[2] == pytest.approx(cfg.BallConfig().radius, abs=REST_TOL)

@pytest.mark.parametrize("vIn", [3.0, 8.0, 15.0])
def testRestitution(makeModel, vIn):
  e = measureRestitution(makeModel(), vIn)
  assert e == pytest.approx(cfg.BallConfig().restitutionFloor, abs=E_TOL)

@pytest.mark.parametrize("eTarget", [0.4, 0.8])
@pytest.mark.parametrize("vIn", [3.0, 15.0])
def testRestitutionOtherE(makeModel, eTarget, vIn):
  model = makeModel(ballCfg=replace(cfg.BallConfig(), restitutionFloor=eTarget))
  assert measureRestitution(model, vIn) == pytest.approx(eTarget, abs=E_TOL)

@pytest.mark.parametrize("timestep", [0.0005, 0.001])
def testRestitutionTimesteps(makeModel, timestep):
  model = makeModel(simCfg=replace(cfg.SimConfig(), timestep=timestep))
  e = measureRestitution(model, 8.0)
  assert e == pytest.approx(cfg.BallConfig().restitutionFloor, abs=E_TOL)

def testRestitution2msNeedsLongerContact(makeModel):
  sim = replace(cfg.SimConfig(), timestep=0.002, contactDuration=0.020)
  e = measureRestitution(makeModel(simCfg=sim), 8.0)
  assert e == pytest.approx(cfg.BallConfig().restitutionFloor, abs=E_TOL)

@pytest.mark.parametrize("eTarget", [0.4, 0.65, 0.8])
def testRestitutionFormulaExact(makeModel, eTarget):
  """With the contact resolved in ~100 steps, the formula itself should be near exact."""
  model = makeModel(
    ballCfg=replace(cfg.BallConfig(), restitutionFloor=eTarget),
    simCfg=replace(cfg.SimConfig(), timestep=0.0001),
  )
  model.opt.gravity[:] = 0      # isolate the contact model
  assert measureRestitution(model, 8.0) == pytest.approx(eTarget, abs=E_TOL_FINE)