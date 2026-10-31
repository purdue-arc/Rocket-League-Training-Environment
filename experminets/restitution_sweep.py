"""Restitution sweep: why the ball bounces too high, and a replacement contact rule.

Run from the project root with the venv active:
  python scripts/restitution_sweep.py

Every experiment compiles the scene once with ball_drop_xml, then overwrites the
ball geom's solref/solimp on the compiled model. MuJoCo lets you change these
parameters after compiling, so no builder changes are needed to try alternatives.

Experiments:
  A  current rule (guide Section 3) over e, speed and timestep
  B  hold the contact fixed, shrink the timestep: does e converge, and to what?
  C  switch off gravity / solimp softening: do they matter?
  D  calibration factor needed for each target e: is 1.25 really a constant?
  E  proposed rule: exact restitution formula + fixed contact duration
"""
import math
from dataclasses import replace

import mujoco

import carSoccer.config as cfg
from carSoccer.engine.sim.scene import ball

SPEEDS = (3.0, 8.0, 15.0)        # m/s, impact speeds
TARGETS = (0.4, 0.65, 0.8)       # target restitution values
FINE_DT = 5e-5                   # s, "practically exact" timestep for reference runs

# ---------- measurement ----------

def compile_scene(timestep):
  """Default ball scene at the given timestep."""
  sim = replace(cfg.SimConfig(), timestep=timestep)
  return mujoco.MjModel.from_xml_string(ball.ball_drop_xml(cfg.BallConfig(), sim))

def ball_geom(model):
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")

def measure(model, v_in, max_time=1.0):
  """Throw the ball down at v_in from 1 mm above the floor.
  
  Returns (e, contact_steps, max_penetration_m).
  """
  data = mujoco.MjData(model)
  radius = model.geom_size[ball_geom(model)][0]
  data.qpos[2] = radius + 0.001
  data.qvel[2] = -v_in
  bounced, peak, steps, pen = False, 0.0, 0, 0.0
  for _ in range(round(max_time / model.opt.timestep)):
    mujoco.mj_step(model, data)
    if data.ncon:
      steps += 1
      pen = max(pen, -data.contact.dist[0])
    vz = data.qvel[2]
    if vz > 0:
      bounced = True
      if vz > peak:
        peak = vz
      elif data.ncon == 0:
        break
    elif bounced:
      break
  if not bounced:
    raise RuntimeError(f"no bounce at v_in={v_in}")
  return peak / v_in, steps, pen

def resting_depth(model, settle_time=4.0):
  """How far the ball sinks into the floor once it has settled, m."""
  data = mujoco.MjData(model)
  data.qpos[2] = 0.5
  for _ in range(round(settle_time / model.opt.timestep)):
    mujoco.mj_step(model, data)
  return model.geom_size[ball_geom(model)][0] - data.qpos[2]

# ---------- contact rules ----------

def textbook_zeta(e):
  """Ideal spring-damper damping ratio for restitution e (no calibration)."""
  le = math.log(e)
  return -le / math.sqrt(math.pi ** 2 + le ** 2)

def current_solref(e, sim):
  """The guide's rule: (time constant, calibrated damping ratio)."""
  return ball.contact_timeconst(sim), ball.damping_ratio(e)

def e_from_zeta(z):
  """Restitution of a spring-damper contact that lets go when its force reaches zero."""
  s = math.sqrt(1 - z * z)
  phi = math.atan2(2 * z * s, 1 - 2 * z * z)
  return math.exp(-z / s * (math.pi - phi))

def zeta_from_e(e):
  """Inverse of e_from_zeta, by bisection. Valid for 0.25 <= e <= 1."""
  lo, hi = 0.0, 1 / math.sqrt(2)
  for _ in range(60):
    mid = (lo + hi) / 2
    if e_from_zeta(mid) > e:
      lo = mid
    else:
      hi = mid
  return (lo + hi) / 2

def proposed_solref(e, duration):
  """Direct-form solref (-stiffness, -damping) for restitution e and contact duration, s."""
  z = zeta_from_e(e)
  w0 = math.pi / (duration * math.sqrt(1 - z * z))    # half a damped oscillation lasts `duration`
  return -w0 * w0, -2 * z * w0

# ---------- experiments ----------

def fmt(values):
  return "  ".join(f"{v:.3f}" for v in values)

def experiment_a():
  print("\nA. Current rule (tc = max(8 dt, 4 ms), zeta x 1.25)")
  print(f"{'dt':>7} {'target':>6} | e at {SPEEDS} m/s     | contact steps @8 m/s")
  for dt in (5e-4, 1e-3, 2e-3):
    model = compile_scene(dt)
    sim = replace(cfg.SimConfig(), timestep=dt)
    for e in TARGETS:
      model.geom_solref[ball_geom(model)] = current_solref(e, sim)
      es = [measure(model, v)[0] for v in SPEEDS]
      print(f"{dt*1e3:>5.1f}ms {e:>6} | {fmt(es)} | {measure(model, 8.0)[1]}")

def experiment_b():
  print("\nB. Contact fixed (tc = 8 ms, textbook zeta for e = 0.65), timestep shrinking, 8 m/s")
  for dt in (1e-3, 5e-4, 2e-4, 1e-4, FINE_DT):
    model = compile_scene(dt)
    model.geom_solref[ball_geom(model)] = (0.008, textbook_zeta(0.65))
    e, steps, _ = measure(model, 8.0)
    print(f"  dt={dt*1e3:>5.2f} ms  e={e:.4f}  contact steps={steps}")

def experiment_c():
  print("\nC. Same contact at fine dt: gravity and solimp switched off, e = 0.65 target, 8 m/s")
  for gravity in (True, False):
    for solimp in (None, (0.99, 0.99, 0.001, 0.5, 2)):
      model = compile_scene(FINE_DT)
      g = ball_geom(model)
      model.geom_solref[g] = (0.008, textbook_zeta(0.65))
      if not gravity:
        model.opt.gravity[:] = 0
      if solimp:
        model.geom_solimp[g] = solimp
      print(f"  gravity={gravity!s:5}  solimp={'near-hard' if solimp else 'default  '}  e={measure(model, 8.0)[0]:.4f}")

def experiment_d():
  print("\nD. Calibration factor c that makes e exact (fine dt, no gravity)")
  model = compile_scene(FINE_DT)
  model.opt.gravity[:] = 0
  g = ball_geom(model)
  for e in (0.3, 0.4, 0.5, 0.65, 0.8, 0.9):
    lo, hi = 0.8, 2.0
    for _ in range(25):
      c = (lo + hi) / 2
      model.geom_solref[g] = (0.008, c * textbook_zeta(e))
      if measure(model, 8.0)[0] > e:
        lo = c
      else:
        hi = c
    print(f"  e={e:<4}  c={c:.3f}")

def experiment_e():
  print("\nE. Proposed rule: exact formula + fixed contact duration T (direct-form solref)")
  print(f"{'T':>5} {'dt':>6} {'N':>4} | " + " | ".join(f"e={e}: {SPEEDS} m/s" for e in TARGETS)
        + " | pen@15  rest")
  for duration in (0.008, 0.010, 0.016):
    for dt in (5e-4, 1e-3, 2e-3):
      model = compile_scene(dt)
      g = ball_geom(model)
      cols = []
      for e in TARGETS:
        model.geom_solref[g] = proposed_solref(e, duration)
        cols.append(fmt(measure(model, v)[0] for v in SPEEDS))
      model.geom_solref[g] = proposed_solref(0.65, duration)
      pen = measure(model, 15.0)[2]
      rest = resting_depth(model)
      print(f"{duration*1e3:>3.0f}ms {dt*1e3:>4.1f}ms {duration/dt:>4.0f} | " + " | ".join(cols)
            + f" | {pen*100:.1f}cm {rest*1e3:.3f}mm")

if __name__ == "__main__":
  experiment_a()
  experiment_b()
  experiment_c()
  experiment_d()
  experiment_e()