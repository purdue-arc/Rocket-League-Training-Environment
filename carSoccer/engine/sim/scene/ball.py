import math, carSoccer.config as cfg

E_MIN = 0.25                               # lowest restitution the formula is checked for

def shell_inertia(mass: float, radius: float) -> float:
  """Moment of inertia of a thin hollow sphere about any axis through its centre. mass in kg and radius in meters."""  
  return (2.0 / 3.0) * mass * radius ** 2

def contact_solref(e: float, sim: cfg.SimConfig) -> tuple[float, float]:
  """Direct-form MuJoCo solref (-stiffness 1/s^2, -damping 1/s) for restitution e and sim.contact_duration in s."""
  steps = sim.contact_duration / sim.timestep
  if steps < sim.min_contact_steps:
    raise ValueError(f"contact lasts {steps:.1f} timesteps, need at least {sim.min_contact_steps}")

  # finds our what dampner and spring work for making ball stay in contact for sim.contact_duration
  # 

  zeta = damping_ratio(e)
  w0 = math.pi / (sim.contact_duration * math.sqrt(1 - zeta ** 2))   # half a damped oscillation lasts contact_duration
  
  return -w0 ** 2, -2 * zeta * w0

def restitution(zeta: float) -> float:
  """Exact restitution of a push-only MuJoCo contact with damping ratio zeta (dimensionless), valid for 0 <= zeta <= 1/sqrt(2)."""
  root = math.sqrt(1 - zeta ** 2)
  phi = math.atan2(2 * zeta * root, 1 - 2 * zeta ** 2)
  
  return math.exp(-zeta / root * (math.pi - phi))

def damping_ratio(e: float) -> float:
  """Converts a coefficient of restitution into MuJoCo's contact damping ratio, by bisection on restitution()."""
  if not (E_MIN <= e <= 1):
    raise ValueError(f"coefficient of restitution must be in [{E_MIN}, 1], got {e!r}")

  # zeta is damping while restitution is bounce, more damping -> less bounce
  # Here user gives us a desired restitution e, and we try to find ideal damping for that desired restitution
  # no closed form inverse exists so we must binary search the answer

  low, high = 0.0, 1 / math.sqrt(2)
  for _ in range(60):
    mid = (low + high) / 2
    if restitution(mid) > e:
      low = mid
    else:
      high = mid

  return (low + high) / 2

def ball_drop_xml(ball_cfg: cfg.BallConfig, sim: cfg.SimConfig, drop_height: float = 1.0) -> str:
  """Builds the MJCF for a single ball above floor, used to test the contact calibration. drop_height is in meters. Returns MJCF XML string."""
  stiffness, damping = contact_solref(ball_cfg.restitution_floor, sim)
  inertia = shell_inertia(ball_cfg.mass, ball_cfg.radius)

  xmlStr = f"""
<mujoco model="ball_drop">
  <option timestep="{sim.timestep:.6g}" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 {drop_height:.6g}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{ball_cfg.mass:.6g}" diaginertia="{inertia:.6g} {inertia:.6g} {inertia:.6g}"/>
      <geom name="ball" type="sphere" size="{ball_cfg.radius:.6g}" solref="{stiffness:.6g} {damping:.6g}" priority="1"/>
    </body>
  </worldbody>
</mujoco>"""

  return xmlStr