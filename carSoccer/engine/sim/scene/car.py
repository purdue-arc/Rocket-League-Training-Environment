import math, mujoco, numpy as np, carSoccer.config as cfg
from dataclasses import dataclass
from carSoccer.engine.sim.scene.arena import WHEEL_CONTYPE, WHEEL_CONAFFINITY

WHEELS = ("FL", "FR", "RL", "RR")            # drive actuator order, 4WD

@dataclass(frozen=True)
class CarGeometry:
  """What the probe measured on the compiled car; every upper layer reads geometry from here, never from CarConfig."""
  wheelbase: float                           # m
  track: float                               # m
  wheelRadius: float                         # m
  spawnHeight: float                         # m, chassis centre height with the wheels on the floor
  maxSteer: float                            # rad, centre steer angle at full lock
  steerSign: float                           # +1 if a positive steer joint angle turns the wheel left
  driveSign: float                           # +1 if a positive drive joint velocity rolls the car forward

def ackermannAngles(steer: float, wheelbase: float, track: float) -> tuple[float, float]:
  """(left, right) wheel angles in rad for a centre steer angle in rad; positive turns left."""
  t = math.tan(steer)
  return math.atan(wheelbase * t / (wheelbase - track / 2 * t)), math.atan(wheelbase * t / (wheelbase + track / 2 * t))

def steerLimit(car: cfg.CarConfig) -> float:
  """Steering joint limit, rad: the inner wheel angle at full lock plus a margin, so sharp turns never hit the stop."""
  return ackermannAngles(car.maxSteer, car.wheelbase, car.track)[0] + 0.05

def carXML(car: cfg.CarConfig) -> str:
  """MJCF of one rectangular car facing +x, chassis centre at the origin. The chassis collides only through <pair>s; sphere wheels only touch the floor."""
  chassisMass = car.mass - 4 * car.wheelMass - 2 * car.knuckleMass
  if chassisMass <= 0:
    raise ValueError(f"wheels and knuckles ({4 * car.wheelMass + 2 * car.knuckleMass} kg) outweigh the car ({car.mass} kg)")
  if car.track / 2 + car.wheelRadius > car.width / 2 or car.wheelbase / 2 + car.wheelRadius > car.length / 2:
    raise ValueError("wheels stick out past the chassis collider in plan view; they would need wall and ball contacts")

  r, lim = car.wheelRadius, steerLimit(car)
  xf, xr, yl, z = car.wheelbase / 2, -car.wheelbase / 2, car.track / 2, -car.axleDrop

  def wheel(name, pos):
    p = " ".join(f"{v:.6g}" for v in pos)
    return f"""
        <body name="Wheel{name}" pos="{p}">
          <joint name="Drive{name}" type="hinge" axis="0 1 0" damping="{car.wheelDamping:.6g}"/>
          <geom name="Wheel{name}" type="sphere" size="{r:.6g}" mass="{car.wheelMass:.6g}" group="3"
                contype="{WHEEL_CONTYPE}" conaffinity="{WHEEL_CONAFFINITY}" condim="3"
                friction="0.01 0.005 0.0001"/>
          <geom name="Tyre{name}" type="cylinder" size="{r:.6g} 0.025" zaxis="0 1 0" mass="0" contype="0" conaffinity="0" material="tyre"/>
        </body>"""

  def knuckle(side, wheelName, y):
    return f"""
      <body name="Knuckle{side}" pos="{xf:.6g} {y:.6g} {z:.6g}">
        <joint name="Steer{side}" type="hinge" axis="0 0 1" range="{-lim:.6g} {lim:.6g}" limited="true"
               damping="{car.steerDamping:.6g}" armature="{car.steerArmature:.6g}"/>
        <inertial pos="0 0 0" mass="{car.knuckleMass:.6g}" diaginertia="1e-5 1e-5 1e-5"/>{wheel(wheelName, (0, 0, 0))}
      </body>"""

  return f"""
<mujoco model="car">
  <compiler angle="radian"/>
  <asset>
    <material name="tyre" rgba="0.08 0.08 0.08 1"/>
    <material name="nose" rgba="1 1 1 1"/>
  </asset>
  <worldbody>
    <body name="Chassis">
      <geom name="Chassis" type="box" size="{car.length / 2:.6g} {car.width / 2:.6g} {car.height / 2:.6g}" mass="{chassisMass:.6g}"
            contype="0" conaffinity="0"/>
      <geom name="Nose" type="box" pos="{car.length / 2 - 0.04:.6g} 0 {car.height / 2:.6g}" size="0.04 {car.width / 4:.6g} 0.005"
            mass="0" contype="0" conaffinity="0" material="nose"/>{knuckle("L", "FL", yl)}{knuckle("R", "FR", -yl)}{wheel("RL", (xr, yl, z))}{wheel("RR", (xr, -yl, z))}
    </body>
  </worldbody>
</mujoco>"""

def carSpec(car: cfg.CarConfig) -> mujoco.MjSpec:
  return mujoco.MjSpec.from_string(carXML(car))

def probeCar(car: cfg.CarConfig, xml: str | None = None) -> CarGeometry:
  """Compile one car alone (carXML(car) unless xml is given) and measure wheel positions and joint axes, which catches rotated frames, wrong units and mirrored axles."""
  model = mujoco.MjModel.from_xml_string(xml or carXML(car))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  def pos(body):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)]

  def axis(joint):
    return data.xaxis[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]

  wheelbase = float(pos("WheelFL")[0] - pos("WheelRL")[0])
  track = float(pos("WheelFL")[1] - pos("WheelFR")[1])
  radius = float(model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "WheelFL")][0])
  steerAxes = np.array([axis(f"Steer{s}") for s in "LR"])
  driveAxes = np.array([axis(f"Drive{w}") for w in WHEELS])
  if not (np.allclose(np.abs(steerAxes[:, 2]), 1) and np.allclose(np.abs(driveAxes[:, 1]), 1)):
    raise ValueError(f"steer axes must be vertical and drive axes lateral, got {steerAxes} and {driveAxes}")
  if not (np.all(np.sign(steerAxes[:, 2]) == np.sign(steerAxes[0, 2])) and np.all(np.sign(driveAxes[:, 1]) == np.sign(driveAxes[0, 1]))):
    raise ValueError("mirrored joint axes within one car; give each joint its own sign")

  return CarGeometry(
    wheelbase=wheelbase,
    track=track,
    wheelRadius=radius,
    spawnHeight=float(-pos("WheelRL")[2] + radius),
    maxSteer=car.maxSteer,
    steerSign=float(np.sign(steerAxes[0, 2])),
    driveSign=float(np.sign(driveAxes[0, 1])),
  )
