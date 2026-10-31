from dataclasses import replace

import mujoco, pytest, numpy as np, math

import carSoccer.config as cfg
from carSoccer.engine.sim.scene import ball, car
from carSoccer.engine.sim.scene.builder import buildScene

E_TOL = 0.05                             # allowed restitution error per pairing

@pytest.fixture(scope="module")
def scene():
  return buildScene(cfg.SceneConfig())

@pytest.fixture
def data(scene):
  return mujoco.MjData(scene.model)

# ---------- helpers ----------

def yaw(quat):
  w, x, y, z = quat
  return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def place(scene, data, carIndex, pos, quat=(1, 0, 0, 0), vel=(0, 0, 0)):
  q, v = scene.layout.carQpos[carIndex], scene.layout.carQvel[carIndex]
  data.qpos[q:q + 3], data.qpos[q + 3:q + 7], data.qvel[v:v + 3] = pos, quat, vel

def placeBall(scene, data, pos, vel=(0, 0, 0), angvel=(0, 0, 0)):
  q, v = scene.layout.ballQpos, scene.layout.ballQvel
  data.qpos[q:q + 3], data.qvel[v:v + 3], data.qvel[v + 3:v + 6] = pos, vel, angvel

def ballVel(scene, data):
  v = scene.layout.ballQvel
  return data.qvel[v:v + 3].copy(), data.qvel[v + 3:v + 6].copy()

def carVel(scene, data, carIndex):
  v = scene.layout.carQvel[carIndex]
  return data.qvel[v:v + 3].copy()

def run(scene, data, seconds):
  for _ in range(round(seconds / scene.model.opt.timestep)):
    mujoco.mj_step(scene.model, data)

def zeroGravityModel(scene):
  model = scene.model.__copy__()
  model.opt.gravity[:] = 0
  return model

# ---------- structure ----------

def testSceneCompiles(scene):
  P = cfg.SceneConfig().carsPerTeam
  assert scene.model.nbody == 2 + 2 * P * 7          # world, ball, and per car: chassis, 2 knuckles, 4 wheels
  assert scene.model.nu == 2 * P * 6                  # 2 steering servos + 4 drive motors per car

def testLayoutShapes(scene):
  L, P = scene.layout, cfg.SceneConfig().carsPerTeam
  assert L.carQpos.shape == L.carQvel.shape == L.touchSensor.shape == (2 * P,)
  assert L.steerQpos.shape == L.steerAct.shape == (2 * P, 2)
  assert L.driveAct.shape == (2 * P, 4)
  assert len(L.carBodies) == 2 * P and all(len(b) == 7 for b in L.carBodies)

def testLayoutIndicesAreTheNamedObjects(scene):
  m, L = scene.model, scene.layout
  free = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "orange2")
  assert L.carQpos[5] == m.jnt_qposadr[free]
  assert m.actuator(int(L.driveAct[3, 2])).name == "orange0DriveRL"

@pytest.mark.parametrize("group", ["ballFloor", "ballWall", "ballCar", "carWall", "carCar"])
def testEachPairingHasItsOwnSolref(scene, group):
  contact = getattr(cfg.ContactsConfig(), group)
  expected = ball.contactSolref(max(contact.restitution, ball.E_MIN), cfg.SimConfig(), contact.duration)
  ids = scene.layout.pairIds[group]
  assert ids
  for i in ids:
    np.testing.assert_allclose(scene.model.pair_solref[i], expected, rtol=1e-5)
    assert scene.model.pair_dim[i] == contact.condim

def testBallFloorRollingFriction(scene):
  i = scene.layout.pairIds["ballFloor"][0]
  assert scene.model.pair_dim[i] == 6
  assert scene.model.pair_friction[i][3] == pytest.approx(cfg.ContactsConfig().ballFloor.rolling)

def testOnlyWheelsAndBallTouchAtRest(scene, data):
  run(scene, data, 0.5)
  P = cfg.SceneConfig().carsPerTeam
  assert data.ncon == 2 * P * 4 + 1                    # every wheel on the floor, plus the ball

# ---------- car probe ----------

def testProbeMatchesConfig():
  c = cfg.CarConfig()
  g = car.probeCar(c)
  assert (g.wheelbase, g.track, g.wheelRadius) == pytest.approx((c.wheelbase, c.track, c.wheelRadius))
  assert g.spawnHeight == pytest.approx(c.axleDrop + c.wheelRadius)
  assert g.steerSign == 1 and g.driveSign == 1

def testProbeCatchesRotatedSteerAxis():
  xml = car.carXML(cfg.CarConfig()).replace('axis="0 0 1"', 'axis="1 0 0"', 1)
  with pytest.raises(ValueError):
    car.probeCar(cfg.CarConfig(), xml)

def testProbeCatchesMirroredAxle():
  xml = car.carXML(cfg.CarConfig()).replace('axis="0 1 0"', 'axis="0 -1 0"', 1)
  with pytest.raises(ValueError):
    car.probeCar(cfg.CarConfig(), xml)

def testSteerLimitCoversInnerWheel():
  c = cfg.CarConfig()
  inner, outer = car.ackermannAngles(c.maxSteer, c.wheelbase, c.track)
  assert inner > c.maxSteer > outer
  assert car.steerLimit(c) == pytest.approx(inner + 0.05)

# ---------- driving ----------

def testCarRestsOnWheels(scene, data):
  run(scene, data, 1.0)
  q = scene.layout.carQpos[0]
  assert data.qpos[q + 2] == pytest.approx(scene.layout.geometry[0].spawnHeight, abs=0.005)

def testThrottleDrivesForward(scene, data):
  q = scene.layout.carQpos[0]
  x0 = data.qpos[q]
  data.ctrl[scene.layout.driveAct[0].numpy()] = 1.0
  run(scene, data, 0.5)
  assert data.qpos[q] - x0 > 0.5                      # blue0 faces +x
  assert abs(data.qpos[q + 1] - (-1.5)) < 0.05        # and drives straight

def testSteerLeftTurnsLeft(scene, data):
  q, L = scene.layout.carQpos[0], scene.layout
  data.ctrl[L.driveAct[0].numpy()] = 0.3
  data.ctrl[L.steerAct[0].numpy()] = car.ackermannAngles(0.4, L.geometry[0].wheelbase, L.geometry[0].track)
  run(scene, data, 1.0)
  assert yaw(data.qpos[q + 3:q + 7]) > 0.3

# ---------- bounce per pairing (zero gravity isolates the contact) ----------

def testBallFloorBounce(scene, data):
  placeBall(scene, data, (0, 0, cfg.BallConfig().radius + 0.001), vel=(0, 0, -8))
  run(scene, data, 0.05)
  assert ballVel(scene, data)[0][2] / 8 == pytest.approx(cfg.ContactsConfig().ballFloor.restitution, abs=E_TOL)

def testBallWallBounce(scene):
  data = mujoco.MjData(model := zeroGravityModel(scene))
  Y, r = cfg.ArenaConfig().width / 2, cfg.BallConfig().radius
  placeBall(scene, data, (0, Y - r - 0.001, cfg.ArenaConfig().wallHeight / 2), vel=(0, 8, 0))
  for _ in range(100):
    mujoco.mj_step(model, data)
  assert -ballVel(scene, data)[0][1] / 8 == pytest.approx(cfg.ContactsConfig().ballWall.restitution, abs=E_TOL)

def testBallCarBounce(scene):
  data = mujoco.MjData(model := zeroGravityModel(scene))
  c, r = cfg.CarConfig(), cfg.BallConfig().radius
  place(scene, data, 0, (-2, -1.5, 0.5))              # both float clear of the floor
  placeBall(scene, data, (-2, -1.5 - c.width / 2 - r - 0.001, 0.5), vel=(0, 6, 0))
  for _ in range(100):
    mujoco.mj_step(model, data)
  rel = ballVel(scene, data)[0][1] - carVel(scene, data, 0)[1]
  assert -rel / 6 == pytest.approx(cfg.ContactsConfig().ballCar.restitution, abs=E_TOL)

def testCarWallBounce(scene):
  data = mujoco.MjData(model := zeroGravityModel(scene))
  Y, c, h = cfg.ArenaConfig().width / 2, cfg.CarConfig(), scene.layout.geometry[0].spawnHeight
  place(scene, data, 2, (-2, Y - c.width / 2 - 0.001, h + 0.02), vel=(0, 3, 0))
  for _ in range(100):
    mujoco.mj_step(model, data)
  assert -carVel(scene, data, 2)[1] / 3 == pytest.approx(cfg.ContactsConfig().carWall.restitution, abs=E_TOL)

def testCarCarBounce(scene):
  data = mujoco.MjData(model := zeroGravityModel(scene))
  c, h = cfg.CarConfig(), scene.layout.geometry[0].spawnHeight
  placeBall(scene, data, (0, 2.5, 1.0))
  place(scene, data, 1, (-c.length / 2 - 0.001, 0, h + 0.02), vel=(2, 0, 0))
  place(scene, data, 4, (c.length / 2 + 0.001, 0, h + 0.02), quat=(0, 0, 0, 1), vel=(-2, 0, 0))
  for _ in range(100):
    mujoco.mj_step(model, data)
  rel = carVel(scene, data, 4)[0] - carVel(scene, data, 1)[0]
  assert rel / 4 == pytest.approx(cfg.ContactsConfig().carCar.restitution, abs=E_TOL)

def testTouchSensorFiresOnBallCarContact(scene, data):
  c, r = cfg.CarConfig(), cfg.BallConfig().radius
  placeBall(scene, data, (-2, -1.5 - c.width / 2 - r - 0.01, r), vel=(0, 4, 0))
  fired = np.zeros(6, dtype=bool)
  for _ in range(200):
    mujoco.mj_step(scene.model, data)
    fired |= data.sensordata[scene.layout.touchSensor.numpy()] > 0
  assert fired.tolist() == [True, False, False, False, False, False]

# ---------- spin ----------

def angledBounce(scene, data, spin=0.0):
  """Ball hits the floor at 45 degrees and 5 m/s per axis; returns (vx, wy) after the bounce."""
  r = cfg.BallConfig().radius
  placeBall(scene, data, (0, 0, r + 0.001), vel=(5, 0, -5), angvel=(0, spin, 0))
  run(scene, data, 0.03)
  v, w = ballVel(scene, data)
  return v[0], w[1]

def testAngledBounceSpinsForward(scene, data):
  vx, wy = angledBounce(scene, data)
  r, k = cfg.BallConfig().radius, 2 / 3                # hollow shell: I = k m r^2
  rolling = 5 / (1 + k)                                # perfect grip
  assert rolling - 0.01 < vx < 5                       # between perfect rolling and no friction
  assert 0 < wy * r < vx + 0.01                        # spins forward, at most rolling

def testBackspinCutsHorizontalSpeed(scene):
  vNone, _ = angledBounce(scene, mujoco.MjData(scene.model))
  vBack, _ = angledBounce(scene, mujoco.MjData(scene.model), spin=-40)
  assert vBack < vNone - 0.5

def testRollingBallSlows(scene, data):
  r = cfg.BallConfig().radius
  placeBall(scene, data, (0, 0, r), vel=(1.5, 0, 0), angvel=(0, 1.5 / r, 0))
  run(scene, data, 1.0)
  assert 0.2 < ballVel(scene, data)[0][0] < 1.4

def testAngularMomentumConservedInFlight(scene, data):
  w0 = np.array([3.0, -7.0, 11.0])
  placeBall(scene, data, (0, 0, 2.5), angvel=w0)
  run(scene, data, 0.3)                                # still airborne
  np.testing.assert_allclose(ballVel(scene, data)[1], w0, rtol=1e-6)
