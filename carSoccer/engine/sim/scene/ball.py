import math, carSoccer.config as cfg

E_MIN = 0.25                               # lowest restitution the formula is checked for

def shellInertia(mass: float, radius: float) -> float:
  """Moment of inertia of a thin hollow sphere about any axis through its centre. mass in kg and radius in meters."""  
  return (2.0 / 3.0) * mass * radius ** 2

def contactSolref(e: float, sim: cfg.SimConfig, duration: float | None = None) -> tuple[float, float]:
  """Direct-form MuJoCo solref (-stiffness 1/s^2, -damping 1/s) for restitution e and a contact lasting duration s (default sim.contactDuration)."""
  duration = sim.contactDuration if duration is None else duration
  steps = duration / sim.timestep
  if steps < sim.minContactSteps:
    raise ValueError(f"contact lasts {steps:.1f} timesteps, need at least {sim.minContactSteps}")

  # finds our what dampner and spring work for making ball stay in contact for duration

  zeta = dampingRatio(e)
  w0 = math.pi / (duration * math.sqrt(1 - zeta ** 2))   # half a damped oscillation lasts duration
  
  return -w0 ** 2, -2 * zeta * w0

def restitution(zeta: float) -> float:
  """Exact restitution of a push-only MuJoCo contact with damping ratio zeta (dimensionless), valid for 0 <= zeta <= 1/sqrt(2)."""
  root = math.sqrt(1 - zeta ** 2)
  phi = math.atan2(2 * zeta * root, 1 - 2 * zeta ** 2)
  
  return math.exp(-zeta / root * (math.pi - phi))

def dampingRatio(e: float) -> float:
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

def ballDropXML(ballCfg: cfg.BallConfig, sim: cfg.SimConfig, dropHeight: float = 1.0) -> str:
  """Builds the MJCF for a single ball above floor, used to test the contact calibration. dropHeight is in meters. Returns MJCF XML string."""
  stiffness, damping = contactSolref(ballCfg.restitutionFloor, sim)
  inertia = shellInertia(ballCfg.mass, ballCfg.radius)

  xmlStr = f"""
<mujoco model="ballDrop">
  <option timestep="{sim.timestep:.6g}" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 {dropHeight:.6g}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{ballCfg.mass:.6g}" diaginertia="{inertia:.6g} {inertia:.6g} {inertia:.6g}"/>
      <geom name="ball" type="sphere" size="{ballCfg.radius:.6g}" solref="{stiffness:.6g} {damping:.6g}" priority="1"/>
    </body>
  </worldbody>
</mujoco>"""

  return xmlStr

def ballBodyXML(ballCfg: cfg.BallConfig, height: float | None = None) -> str:
  """MJCF body for the match ball at the centre spot. Its contacts come only from explicit <pair>s, so the geom collides with nothing on its own."""
  inertia = shellInertia(ballCfg.mass, ballCfg.radius)
  height = ballCfg.radius if height is None else height

  return f"""
    <body name="ball" pos="0 0 {height:.6g}">
      <freejoint name="ball"/>
      <inertial pos="0 0 0" mass="{ballCfg.mass:.6g}" diaginertia="{inertia:.6g} {inertia:.6g} {inertia:.6g}"/>
      <geom name="ball" type="sphere" size="{ballCfg.radius:.6g}" contype="0" conaffinity="0" material="ball"/>
    </body>"""
