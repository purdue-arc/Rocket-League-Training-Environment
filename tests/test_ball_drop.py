from dataclasses import replace

import mujoco
import numpy as np
import pytest

import carSoccer.config as cfg

GRAVITY = 9.81                           # m/s^2
E_TOL = 0.04                             # allowed restitution error with >= 10 steps per contact (sweep experiment E)
E_TOL_FINE = 0.01                        # allowed error when the contact is resolved in ~100 steps
REST_TOL = 0.0005                        # m; resting ball sinks well under 0.1 mm (sweep experiment E)

def measure_restitution(model, v_in, max_time=1.0):
  """Throw the ball down at v_in (m/s) from just above the floor; return bounce speed / v_in."""
  data = mujoco.MjData(model)
  g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
  radius = model.geom_size[g][0]
  
  data.qpos[2] = radius + 0.001      # 1 mm gap: gravity barely changes v_in before impact
  data.qvel[2] = -v_in
  
  bounced, peak = False, 0.0
  for _ in range(round(max_time / model.opt.timestep)):
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
  assert bounced, f"ball never bounced within {max_time} s at v_in={v_in}"
  return peak / v_in

def test_gravity(model):
  np.testing.assert_allclose(model.opt.gravity, [0.0, 0.0, -GRAVITY], rtol=1e-6)

def test_free_fall(make_model):
  model = make_model(drop_height=2.0)
  data = mujoco.MjData(model)
  for _ in range(round(0.3 / model.opt.timestep)):     # lands after ~0.6 s, so still airborne
    mujoco.mj_step(model, data)
  t = data.time
  assert data.qvel[2] == pytest.approx(-GRAVITY * t, rel=1e-4)
  # position drifts ~g*t*dt/2 from the exact parabola due to time stepping
  assert data.qpos[2] == pytest.approx(2.0 - 0.5 * GRAVITY * t**2, abs=5e-3)

def test_ball_comes_to_rest(make_model):
  model = make_model(drop_height=0.5)
  data = mujoco.MjData(model)
  for _ in range(round(5.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
  assert np.linalg.norm(data.qvel[:3]) < 1e-2
  assert data.qpos[2] == pytest.approx(cfg.BallConfig().radius, abs=REST_TOL)

@pytest.mark.parametrize("v_in", [3.0, 8.0, 15.0])
def test_restitution(make_model, v_in):
  e = measure_restitution(make_model(), v_in)
  assert e == pytest.approx(cfg.BallConfig().restitution_floor, abs=E_TOL)

@pytest.mark.parametrize("e_target", [0.4, 0.8])
@pytest.mark.parametrize("v_in", [3.0, 15.0])
def test_restitution_other_e(make_model, e_target, v_in):
  model = make_model(ball_cfg=replace(cfg.BallConfig(), restitution_floor=e_target))
  assert measure_restitution(model, v_in) == pytest.approx(e_target, abs=E_TOL)

@pytest.mark.parametrize("timestep", [0.0005, 0.001])
def test_restitution_timesteps(make_model, timestep):
  model = make_model(sim_cfg=replace(cfg.SimConfig(), timestep=timestep))
  e = measure_restitution(model, 8.0)
  assert e == pytest.approx(cfg.BallConfig().restitution_floor, abs=E_TOL)

def test_restitution_2ms_needs_longer_contact(make_model):
  sim = replace(cfg.SimConfig(), timestep=0.002, contact_duration=0.020)
  e = measure_restitution(make_model(sim_cfg=sim), 8.0)
  assert e == pytest.approx(cfg.BallConfig().restitution_floor, abs=E_TOL)

@pytest.mark.parametrize("e_target", [0.4, 0.65, 0.8])
def test_restitution_formula_exact(make_model, e_target):
  """With the contact resolved in ~100 steps, the formula itself should be near exact."""
  model = make_model(
    ball_cfg=replace(cfg.BallConfig(), restitution_floor=e_target),
    sim_cfg=replace(cfg.SimConfig(), timestep=0.0001),
  )
  model.opt.gravity[:] = 0      # isolate the contact model
  assert measure_restitution(model, 8.0) == pytest.approx(e_target, abs=E_TOL_FINE)