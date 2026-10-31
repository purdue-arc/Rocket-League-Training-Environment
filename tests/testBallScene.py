from dataclasses import replace

import mujoco, pytest, numpy as np, math

import carSoccer.config as cfg
from carSoccer.engine.sim.scene import ball

# ---------- helpers -----------------

def bodyId(model, name):
  i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
  assert i >= 0, f"no body named {name!r}"
  return i

def geomId(model, name):
  i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
  assert i >= 0, f"no geom named {name!r}"
  return i

def solrefToContact(stiff, damp):
  """Recover (duration s, zeta) from a direct-form solref."""
  w0 = math.sqrt(-stiff)
  zeta = -damp / (2 * w0)
  return math.pi / (w0 * math.sqrt(1 - zeta ** 2)), zeta

# ---------- inertia ----------

def testShellInertiaValue():
  assert ball.shellInertia(0.45, 0.2032) == pytest.approx(0.01239, rel=1e-3)

def testShellInertiaScalesWithMass():
  i1 = ball.shellInertia(0.45, 0.2032)
  i2 = ball.shellInertia(0.90, 0.2032)
  assert i2 == pytest.approx(2 * i1)

# ---------- restitution <-> damping ratio ----------

def testRestitutionNoDamping():
  assert ball.restitution(0.0) == pytest.approx(1.0)

def testDampingRatioValue():
  assert ball.dampingRatio(0.65) == pytest.approx(0.1499, abs=1e-3)

def testDampingRatioPerfectBounce():
  assert ball.dampingRatio(1.0) == pytest.approx(0.0, abs=1e-9)

@pytest.mark.parametrize("e", [0.25, 0.4, 0.65, 0.8, 0.95])
def testDampingRatioRoundTrip(e):
  assert ball.restitution(ball.dampingRatio(e)) == pytest.approx(e, abs=1e-6)

def testDampingRatioDecreasing():
  zLow, zMid, zHigh = (ball.dampingRatio(e) for e in (0.4, 0.65, 0.9))
  assert zLow > zMid > zHigh

@pytest.mark.parametrize("e", [0.0, 0.2, -0.1, 1.5])
def testDampingRatioInvalid(e):
  with pytest.raises(ValueError):
    ball.dampingRatio(e)

# ---------- contact solref ----------
 
def testContactSolrefIsDirectForm():
  stiff, damp = ball.contactSolref(0.65, cfg.SimConfig())
  assert stiff < 0 and damp < 0

@pytest.mark.parametrize("e", [0.4, 0.65, 1.0])
def testContactSolrefRoundTrip(e):
  sim = cfg.SimConfig()
  duration, zeta = solrefToContact(*ball.contactSolref(e, sim))
  assert duration == pytest.approx(sim.contactDuration)
  assert zeta == pytest.approx(ball.dampingRatio(e))

def testContactSolrefRejectsCoarseTimestep():
  sim = replace(cfg.SimConfig(), timestep=0.002)        # 10 ms / 2 ms = 5 steps < 10
  with pytest.raises(ValueError):
    ball.contactSolref(0.65, sim)

def testContactSolrefMinimumStepsBoundary():
  sim = cfg.SimConfig()
  sim = replace(sim, timestep=sim.contactDuration / sim.minContactSteps)
  ball.contactSolref(0.65, sim)                         # exactly the minimum: allowed

# ---------- MJCF builder ----------
 
def testXmlCompiles(model):
  assert model.nbody == 2          # world + ball

def testTimestepMatches(model):
  assert model.opt.timestep == pytest.approx(cfg.SimConfig().timestep)

def testBallMass(model):
  assert model.body_mass[bodyId(model, "ball")] == pytest.approx(cfg.BallConfig().mass)

def testBallInertia(model):
  b = cfg.BallConfig()
  expected = ball.shellInertia(b.mass, b.radius)
  np.testing.assert_allclose(model.body_inertia[bodyId(model, "ball")], [expected] * 3, rtol=1e-5)

def testBallRadius(model):
  assert model.geom_size[geomId(model, "ball")][0] == pytest.approx(cfg.BallConfig().radius)

def testBallStartHeight(makeModel):
  model = makeModel(dropHeight=0.7)
  data = mujoco.MjData(model)
  assert data.qpos[2] == pytest.approx(0.7)    # free joint qpos = [x, y, z, qw, qx, qy, qz]

def testBallSolref(model):
  expected = ball.contactSolref(cfg.BallConfig().restitutionFloor, cfg.SimConfig())
  np.testing.assert_allclose(model.geom_solref[geomId(model, "ball")], expected, rtol=1e-5)

def testBallPriority(model):
  assert model.geom_priority[geomId(model, "ball")] > model.geom_priority[geomId(model, "floor")]

def testConfigChangePropagates(makeModel):
  b = replace(cfg.BallConfig(), radius=0.1, mass=0.3, restitutionFloor=0.5)
  model = makeModel(ballCfg=b)
  g = geomId(model, "ball")
  assert model.geom_size[g][0] == pytest.approx(0.1)
  assert model.body_mass[bodyId(model, "ball")] == pytest.approx(0.3)
  np.testing.assert_allclose(model.geom_solref[g], ball.contactSolref(0.5, cfg.SimConfig()), rtol=1e-5)