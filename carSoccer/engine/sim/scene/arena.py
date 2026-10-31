import math, carSoccer.config as cfg
from carSoccer.engine.sim.scene import ball

# collision bits: the floor collides on its own only with wheels; everything else uses explicit <pair>s
FLOOR_CONTYPE, FLOOR_CONAFFINITY = 1, 0
WHEEL_CONTYPE, WHEEL_CONAFFINITY = 0, 1

def quatZ(angle: float) -> str:
  """MJCF quat (w x y z) for a rotation of angle rad about +z."""
  return f"{math.cos(angle / 2):.6g} 0 0 {math.sin(angle / 2):.6g}"

def boxGeom(name: str, pos, half, material: str, angle: float = 0.0) -> str:
  p = " ".join(f"{v:.6g}" for v in pos)
  s = " ".join(f"{v:.6g}" for v in half)
  return f'    <geom name="{name}" type="box" pos="{p}" size="{s}" quat="{quatZ(angle)}" contype="0" conaffinity="0" material="{material}"/>\n'

def wallGeoms(arena: cfg.ArenaConfig) -> list[tuple]:
  """(name, pos, half sizes, material, yaw) of every wall, chamfer and goal piece. All are boxes outside the playing area."""
  X, Y, c, t, h = arena.length / 2, arena.width / 2, arena.chamfer, arena.wallThickness, arena.wallHeight
  G, gh, gd = arena.goalWidth / 2, arena.goalHeight, arena.goalDepth
  pieces = []

  for sy, side in ((1, "Left"), (-1, "Right")):
    pieces.append((f"wallSide{side}", (0, sy * (Y + t / 2), h / 2), (X - c + t, t / 2, h / 2), "wall", 0.0))

  # end walls either side of each goal mouth, extended into the chamfer so there is no gap at the joint
  for sx, end in ((1, "Orange"), (-1, "Blue")):
    half = (Y - c + t - G) / 2
    for sy, side in ((1, "Left"), (-1, "Right")):
      pieces.append((f"wallEnd{end}{side}", (sx * (X + t / 2), sy * (G + half), h / 2), (t / 2, half, h / 2), "wall", 0.0))

  # 45 degree corner pieces from (X - c, Y) to (X, Y - c), pushed outward by half their thickness
  off = c / 2 - t / (2 * math.sqrt(2))
  for sx, end in ((1, "Orange"), (-1, "Blue")):
    for sy, side in ((1, "Left"), (-1, "Right")):
      pos = (sx * (X - off), sy * (Y - off), h / 2)
      pieces.append((f"chamfer{end}{side}", pos, (c / math.sqrt(2) + t / 2, t / 2, h / 2), "wall", math.atan2(-sy, sx)))

  # goals: the ball scores once it is fully past x = +-X between the posts
  for sx, end in ((1, "Orange"), (-1, "Blue")):
    mat = f"goal{end}"
    pieces.append((f"goal{end}Back", (sx * (X + gd + t / 2), 0, gh / 2), (t / 2, G + t, gh / 2), mat, 0.0))
    for sy, side in ((1, "Left"), (-1, "Right")):
      pieces.append((f"goal{end}{side}", (sx * (X + gd / 2), sy * (G + t / 2), gh / 2), (gd / 2, t / 2, gh / 2), mat, 0.0))
    pieces.append((f"goal{end}Roof", (sx * (X + gd / 2 + t / 2), 0, gh + t / 2), (gd / 2 + t / 2, G + t, t / 2), mat, 0.0))

  return pieces

def arenaXML(arena: cfg.ArenaConfig, ballCfg: cfg.BallConfig, sim: cfg.SimConfig) -> str:
  """World MJCF: options, assets, floor, walls, goals, field markings, lights, cameras and the ball. Cars are attached later."""
  X, Y = arena.length / 2, arena.width / 2
  walls = "".join(boxGeom(*piece) for piece in wallGeoms(arena))

  return f"""
<mujoco model="carSoccer">
  <compiler angle="radian"/>
  <option timestep="{sim.timestep:.6g}" integrator="implicitfast"/>
  <statistic center="0 0 0" extent="6"/>
  <visual>
    <global offwidth="1280" offheight="720" azimuth="90" elevation="-35"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.35 0.45 0.6" rgb2="0.05 0.05 0.1" width="32" height="32"/>
    <texture name="turf" type="2d" builtin="checker" rgb1="0.16 0.42 0.2" rgb2="0.14 0.38 0.18" width="32" height="32"/>
    <material name="turf" texture="turf" texrepeat="12 8" reflectance="0.05"/>
    <material name="line" rgba="0.9 0.9 0.9 1"/>
    <material name="wall" rgba="0.75 0.85 1 0.35"/>
    <material name="goalBlue" rgba="0.2 0.4 1 0.6"/>
    <material name="goalOrange" rgba="1 0.55 0.1 0.6"/>
    <material name="ball" rgba="0.95 0.95 0.95 1" specular="0.4"/>
    <material name="carBlue" rgba="0.15 0.35 0.95 1"/>
    <material name="carOrange" rgba="0.95 0.5 0.1 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <camera name="broadcast" pos="0 -7.5 5.5" xyaxes="1 0 0 0 0.591 0.806"/>
    <camera name="top" pos="0 0 11" xyaxes="1 0 0 0 1 0"/>
    <geom name="floor" type="plane" size="{X + arena.goalDepth + 2:.6g} {Y + 2:.6g} 0.1" material="turf"
          contype="{FLOOR_CONTYPE}" conaffinity="{FLOOR_CONAFFINITY}" friction="{arena.floorFriction:.6g} 0.005 0.0001"/>
    <geom name="lineHalfway" type="box" pos="0 0 0.001" size="0.02 {Y:.6g} 0.001" material="line" contype="0" conaffinity="0"/>
    <geom name="lineGoalBlue" type="box" pos="{-X:.6g} 0 0.001" size="0.02 {arena.goalWidth / 2:.6g} 0.001" material="goalBlue" contype="0" conaffinity="0"/>
    <geom name="lineGoalOrange" type="box" pos="{X:.6g} 0 0.001" size="0.02 {arena.goalWidth / 2:.6g} 0.001" material="goalOrange" contype="0" conaffinity="0"/>
{walls}{ball.ballBodyXML(ballCfg)}
  </worldbody>
</mujoco>"""
