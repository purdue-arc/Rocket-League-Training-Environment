from dataclasses import replace

import mujoco, pytest, numpy as np, math

import carSoccer.config as cfg
from carSoccer.engine.sim.scene import ball

# ---------- helpers -----------------

def body_id(model, name):
  i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
  assert i >= 0, f"no body named {name!r}"
  return i

def geom_id(model, name):
  i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
  assert i >= 0, f"no geom named {name!r}"
  return i

def solref_to_contact(stiff, damp):
  """Recover (duration s, zeta) from a direct-form solref."""
  w0 = math.sqrt(-stiff)
  zeta = -damp / (2 * w0)
  return math.pi / (w0 * math.sqrt(1 - zeta ** 2)), zeta

# ---------- inertia ----------

def test_shell_inertia_value():
  assert ball.shell_inertia(0.45, 0.2032) == pytest.approx(0.01239, rel=1e-3)

def test_shell_inertia_scales_with_mass():
  i1 = ball.shell_inertia(0.45, 0.2032)
  i2 = ball.shell_inertia(0.90, 0.2032)
  assert i2 == pytest.approx(2 * i1)

# ---------- restitution <-> damping ratio ----------

def test_restitution_no_damping():
  assert ball.restitution(0.0) == pytest.approx(1.0)

def test_damping_ratio_value():
  assert ball.damping_ratio(0.65) == pytest.approx(0.1499, abs=1e-3)

def test_damping_ratio_perfect_bounce():
  assert ball.damping_ratio(1.0) == pytest.approx(0.0, abs=1e-9)

@pytest.mark.parametrize("e", [0.25, 0.4, 0.65, 0.8, 0.95])
def test_damping_ratio_round_trip(e):
  assert ball.restitution(ball.damping_ratio(e)) == pytest.approx(e, abs=1e-6)

def test_damping_ratio_decreasing():
  z_low, z_mid, z_high = (ball.damping_ratio(e) for e in (0.4, 0.65, 0.9))
  assert z_low > z_mid > z_high

@pytest.mark.parametrize("e", [0.0, 0.2, -0.1, 1.5])
def test_damping_ratio_invalid(e):
  with pytest.raises(ValueError):
    ball.damping_ratio(e)

# ---------- contact solref ----------
 
def test_contact_solref_is_direct_form():
  stiff, damp = ball.contact_solref(0.65, cfg.SimConfig())
  assert stiff < 0 and damp < 0

@pytest.mark.parametrize("e", [0.4, 0.65, 1.0])
def test_contact_solref_round_trip(e):
  sim = cfg.SimConfig()
  duration, zeta = solref_to_contact(*ball.contact_solref(e, sim))
  assert duration == pytest.approx(sim.contact_duration)
  assert zeta == pytest.approx(ball.damping_ratio(e))

def test_contact_solref_rejects_coarse_timestep():
  sim = replace(cfg.SimConfig(), timestep=0.002)        # 10 ms / 2 ms = 5 steps < 10
  with pytest.raises(ValueError):
    ball.contact_solref(0.65, sim)

def test_contact_solref_minimum_steps_boundary():
  sim = cfg.SimConfig()
  sim = replace(sim, timestep=sim.contact_duration / sim.min_contact_steps)
  ball.contact_solref(0.65, sim)                         # exactly the minimum: allowed

# ---------- MJCF builder ----------
 
def test_xml_compiles(model):
  assert model.nbody == 2          # world + ball

def test_timestep_matches(model):
  assert model.opt.timestep == pytest.approx(cfg.SimConfig().timestep)

def test_ball_mass(model):
  assert model.body_mass[body_id(model, "ball")] == pytest.approx(cfg.BallConfig().mass)

def test_ball_inertia(model):
  b = cfg.BallConfig()
  expected = ball.shell_inertia(b.mass, b.radius)
  np.testing.assert_allclose(model.body_inertia[body_id(model, "ball")], [expected] * 3, rtol=1e-5)

def test_ball_radius(model):
  assert model.geom_size[geom_id(model, "ball")][0] == pytest.approx(cfg.BallConfig().radius)

def test_ball_start_height(make_model):
  model = make_model(drop_height=0.7)
  data = mujoco.MjData(model)
  assert data.qpos[2] == pytest.approx(0.7)    # free joint qpos = [x, y, z, qw, qx, qy, qz]

def test_ball_solref(model):
  expected = ball.contact_solref(cfg.BallConfig().restitution_floor, cfg.SimConfig())
  np.testing.assert_allclose(model.geom_solref[geom_id(model, "ball")], expected, rtol=1e-5)

def test_ball_priority(model):
  assert model.geom_priority[geom_id(model, "ball")] > model.geom_priority[geom_id(model, "floor")]

def test_config_change_propagates(make_model):
  b = replace(cfg.BallConfig(), radius=0.1, mass=0.3, restitution_floor=0.5)
  model = make_model(ball_cfg=b)
  g = geom_id(model, "ball")
  assert model.geom_size[g][0] == pytest.approx(0.1)
  assert model.body_mass[body_id(model, "ball")] == pytest.approx(0.3)
  np.testing.assert_allclose(model.geom_solref[g], ball.contact_solref(0.5, cfg.SimConfig()), rtol=1e-5)