import numpy as np, torch, carSoccer.config as cfg
from torch import Tensor
from carSoccer.engine import quat
from carSoccer.engine.base import EngineCaps, GameState, StepEvents, emptyEvents, emptyGameState, maskedAssign
from carSoccer.engine.sim import physics, rules
from carSoccer.engine.sim.backend.cpuRollout import CpuRolloutBackend
from carSoccer.engine.sim.scene import ball as ballScene
from carSoccer.engine.sim.scene.builder import Scene
from carSoccer.engine.sim.scene.contacts import PAIRINGS, solrefBatch
from carSoccer.engine.sim.sensors import SensorModel

def nominalParams(sceneCfg: cfg.SceneConfig) -> dict[str, list]:
  """Per-world physics parameters of the compiled model, before any randomisation (absolute values, not scales)."""
  pairings = [getattr(sceneCfg.contacts, name) for name in PAIRINGS]
  return {
    "floorFriction": sceneCfg.arena.floorFriction,                          # sliding coefficient
    "ballMass": sceneCfg.ball.mass,                                         # kg
    "restitution": [c.restitution for c in pairings],                       # per pairing, PAIRINGS order
    "contactFriction": [c.friction for c in pairings],                      # per pairing
  }

class SimEngine:
  """Scene + backend + car and ball physics. Advances N worlds one control tick per step(); never resets a world on its own."""

  def __init__(self, scene: Scene, sceneCfg: cfg.SceneConfig, game: cfg.GameConfig, numWorlds: int,
               nthread: int | None = None, backend=None, generator: torch.Generator | None = None):
    self.scene, self.sceneCfg, self.game, self.numWorlds = scene, sceneCfg, game, numWorlds
    self.backend = backend or CpuRolloutBackend(scene.model, numWorlds, nthread)
    self.device = self.backend.device
    self.capabilities = EngineCaps(teleport=True, trueState=True, parallelWorlds=True)
    L, N, dev = scene.layout, numWorlds, self.device
    self.P = P = L.carsPerTeam

    self.dt = 1 / game.controlHz
    self.nsub = round(self.dt / sceneCfg.sim.timestep)
    if abs(self.nsub * sceneCfg.sim.timestep - self.dt) > 1e-9:
      raise ValueError(f"control period {self.dt} s is not a whole number of {sceneCfg.sim.timestep} s timesteps")

    # gather indices into qpos / qvel, built once
    r3, r4 = torch.arange(3), torch.arange(4)
    self.carPosIdx, self.carQuatIdx = L.carQpos[:, None] + r3, L.carQpos[:, None] + 3 + r4
    self.carVelIdx, self.carAngIdx = L.carQvel[:, None] + r3, L.carQvel[:, None] + 3 + r3
    self.ballPosIdx, self.ballQuatIdx = L.ballQpos + r3, L.ballQpos + 3 + r4
    self.ballVelIdx, self.ballAngIdx = L.ballQvel + r3, L.ballQvel + 3 + r3
    self.steerIdx = L.steerQpos.flatten()

    # car geometry per variant (V, 4): wheelbase, track, max steer, spawn height; gathered per world by refreshGeometry
    self.geometryTable = torch.tensor([[g.wheelbase, g.track, g.maxSteer, g.spawnHeight] for g in L.geometry], device=dev)
    self.spawnHeights = self.geometryTable[:, 3]
    self.steerSign = L.geometry[0].steerSign
    # parameter packs stacked over variants, (V, ...) float64, the base for every per-world model write
    self.packs = {k: torch.from_numpy(np.stack([p[k] for p in scene.variants])) for k in scene.variants[0]}

    # actuator state and the latency ring buffer
    R = game.maxLatencyTicks + 1
    self.history = torch.zeros(N, R, 2 * P, 2, device=dev)
    self.ringPos = 0
    self.worldIdx = torch.arange(N, device=dev)
    self.throttle = torch.zeros(N, 2 * P, device=dev)
    self.steerAngle = torch.zeros(N, 2 * P, device=dev)
    self.goalLatch = torch.zeros(N, dtype=torch.bool, device=dev)
    self.ctrl = torch.zeros(N, scene.model.nu, dtype=torch.float64, device=dev)
    self.xfrc = torch.zeros(N, scene.model.nbody, 6, dtype=torch.float64, device=dev)

    self.generator = generator or torch.Generator(device=dev).manual_seed(0)
    self.sensors = SensorModel(N, P, game.sensors, self.generator, dev)
    self.state = emptyGameState(N, P, dev)
    params = self.state["params"]
    params["latency"].fill_(game.latencyTicks)
    for key, value in nominalParams(sceneCfg).items():
      params[key] = torch.tensor(value, device=dev).expand(N, *torch.tensor(value).shape).clone()
    params["motorStrength"] = torch.ones(N, 2 * P, device=dev)              # scale on stall torque, per car
    params["steerTrim"] = torch.zeros(N, 2 * P, device=dev)                 # rad, per car
    self.refreshGeometry()
    self.readState()
    self.state["measured"] = self.sensors.reset(torch.ones(N, dtype=torch.bool, device=dev), self.state)

  # ---------- one control tick ----------

  def step(self, carActions: Tensor) -> StepEvents:
    L, game, dt = self.scene.layout, self.game, self.dt

    # 1. latency: each world reads the command sent `latency` ticks ago
    self.ringPos = (self.ringPos + 1) % self.history.shape[1]
    self.history[:, self.ringPos] = carActions.clamp(-1, 1)
    slot = (self.ringPos - self.state["params", "latency"]) % self.history.shape[1]
    throttle, steer = self.history[self.worldIdx, slot].unbind(-1)

    # 2-3. motor and steering servo, then the Ackermann split
    self.throttle = physics.motorStep(throttle, self.throttle, game, dt)
    target = steer * self.maxSteer + self.state["params", "steerTrim"]
    self.steerAngle = physics.servoStep(target, self.steerAngle, game, dt)
    left, right = physics.ackermannSplit(self.steerAngle, self.wheelbase, self.track)
    self.ctrl[:, L.steerAct[:, 0]] = left.double()
    self.ctrl[:, L.steerAct[:, 1]] = right.double()
    self.ctrl[:, L.driveAct] = self.throttle[..., None].double().expand(-1, -1, L.driveAct.shape[1])

    # 4. ball aerodynamics, applied at the ball only
    ball = self.state["ball"]
    self.xfrc[:, L.ballBody, :3] = physics.ballAero(ball["vel"], ball["angvel"], self.sceneCfg.ball, game).double()

    # 5-6. physics, then one gather from qpos / qvel
    self.backend.step(self.ctrl, self.xfrc, self.nsub)
    self.readState()

    # 7. rules
    events = emptyEvents(self.numWorlds, self.P, self.device)
    inGoal = rules.ballInGoal(self.state["ball", "pos"], self.sceneCfg.arena, self.sceneCfg.ball.radius)
    goal, self.goalLatch = rules.latchGoals(inGoal, self.goalLatch)
    events["goal"] = goal
    events["ballTouch"] = self.backend.sensordata[:, L.touchSensor] > 0
    events["ballOut"] = rules.ballOut(self.state["ball", "pos"], self.sceneCfg.arena, game.ballOutMargin)
    events["flipped"] = isFlipped = rules.flipped(self.state["car", "quat"], game.flippedUp)
    upright = self.state["car", "uprightTime"]
    self.state["car", "uprightTime"] = torch.where(isFlipped, upright + dt, torch.zeros_like(upright))
    self.state["score"] += goal.long()
    self.state["clock"] += 1
    self.recoverFlipped(self.state["car", "uprightTime"] > game.flipRecoverSeconds)

    # 8. sensors
    self.state["measured"] = self.sensors.measure(self.state)
    return events

  def readState(self) -> None:
    qpos, qvel = self.backend.qpos, self.backend.qvel
    car, ball = self.state["car"], self.state["ball"]
    car["pos"], car["quat"] = qpos[:, self.carPosIdx].float(), qpos[:, self.carQuatIdx].float()
    car["vel"] = qvel[:, self.carVelIdx].float()
    car["angvel"] = quat.rotate(car["quat"], qvel[:, self.carAngIdx].float())       # free joint angular velocity is body frame
    car["steer"] = self.steerSign * qpos[:, self.steerIdx].view(self.numWorlds, -1, 2).mean(-1).float()
    ball["pos"], ballQuat = qpos[:, self.ballPosIdx].float(), qpos[:, self.ballQuatIdx].float()
    ball["vel"], ball["angvel"] = qvel[:, self.ballVelIdx].float(), quat.rotate(ballQuat, qvel[:, self.ballAngIdx].float())

  # ---------- writes ----------

  def writeBodies(self, carMask: Tensor, ballMask: Tensor, target: GameState, resetJoints: Tensor) -> None:
    """Teleport masked cars (N, 2P) and balls (N,) to target; worlds in resetJoints (N,) also get straight wheels and no spin."""
    qpos, qvel = self.backend.qpos.clone(), self.backend.qvel.clone()
    car, ball = target["car"], target["ball"]
    cm, bm, jm = carMask[..., None], ballMask[:, None], resetJoints[:, None]

    def put(buf, idx, mask, value):
      buf[:, idx] = torch.where(mask, value.double(), buf[:, idx])

    qpos = torch.where(jm, self.backend.qpos0.expand_as(qpos), qpos)
    qvel = torch.where(jm, torch.zeros_like(qvel), qvel)
    put(qpos, self.carPosIdx, cm, car["pos"])
    put(qpos, self.carQuatIdx, cm, car["quat"])
    put(qvel, self.carVelIdx, cm, car["vel"])
    put(qvel, self.carAngIdx, cm, quat.rotateInverse(car["quat"], car["angvel"]))
    put(qpos, self.ballPosIdx, bm, ball["pos"])
    put(qpos, self.ballQuatIdx, bm, torch.tensor([1.0, 0, 0, 0], device=self.device).expand(self.numWorlds, 4))
    put(qvel, self.ballVelIdx, bm, ball["vel"])
    put(qvel, self.ballAngIdx, bm, ball["angvel"])           # ball quat is identity, so body frame = world frame
    self.backend.writeState(carMask.any(-1) | ballMask, qpos, qvel)

  def setState(self, mask: Tensor, target: GameState) -> None:
    """Teleport masked worlds to target and clear their actuator state, latency history and goal latch."""
    carMask = mask[:, None].expand(-1, 2 * self.P)
    self.writeBodies(carMask, mask, target, resetJoints=mask)
    maskedAssign(self.state, target.exclude("car", "ball", "measured", "params"), mask)
    maskedAssign(self.state["car"], target["car"].select("uprightTime"), mask)
    m2 = mask[:, None]
    self.throttle = torch.where(m2, torch.zeros_like(self.throttle), self.throttle)
    self.steerAngle = torch.where(m2, torch.zeros_like(self.steerAngle), self.steerAngle)
    self.history = torch.where(mask[:, None, None, None], torch.zeros_like(self.history), self.history)
    self.goalLatch = self.goalLatch & ~mask
    self.readState()
    maskedAssign(self.state["measured"], self.sensors.reset(mask, self.state), mask)

  def recoverFlipped(self, need: Tensor) -> None:
    """Set cars flipped for too long back on their wheels, in place, keeping position and heading."""
    car = self.state["car"]
    target = self.state.clone()
    pos = car["pos"].clone()
    pos[..., 2] = self.worldSpawnHeight[:, None] + 0.02
    target["car", "pos"] = pos
    target["car", "quat"] = quat.fromYaw(quat.yaw(car["quat"]))
    target["car", "vel"] = torch.zeros_like(car["vel"])
    target["car", "angvel"] = torch.zeros_like(car["angvel"])
    noWorld = torch.zeros(self.numWorlds, dtype=torch.bool, device=self.device)
    self.writeBodies(need, noWorld, target, resetJoints=noWorld)
    car["uprightTime"] = torch.where(need, torch.zeros_like(car["uprightTime"]), car["uprightTime"])
    self.readState()

  def setParams(self, mask: Tensor, params) -> None:
    """Per-world physics parameters for masked worlds (a dict or TensorDict of any of state.params' keys: variant,
    latency, floorFriction, ballMass, restitution, contactFriction, motorStrength, steerTrim). The model fields they
    imply are rebuilt from the world's variant pack and written into that world's model."""
    current = self.state["params"]
    for key in params.keys():
      value = params[key]
      if key == "latency":
        value = value.clamp(0, self.game.maxLatencyTicks)
      if key == "variant":
        value = value.clamp(0, len(self.scene.variants) - 1)
      old = current[key]
      current[key] = torch.where(mask.view(-1, *[1] * (old.dim() - 1)), value.to(old.dtype), old)
    self.backend.setParams(mask, self.modelFields())
    self.refreshGeometry()

  def modelFields(self) -> dict[str, Tensor]:
    """Model fields (N, ...) float64 for every world from its variant pack and drawn physics parameters."""
    L, p = self.scene.layout, self.state["params"]
    fields = {k: table[p["variant"]].clone() for k, table in self.packs.items()}
    m = p["ballMass"].double()
    fields["body_mass"][:, L.ballBody] = m
    fields["body_subtreemass"][:, L.ballBody] = m
    fields["body_inertia"][:, L.ballBody] = ballScene.shellInertia(m, self.sceneCfg.ball.radius)[:, None]
    fields["geom_friction"][:, L.floorGeom, 0] = p["floorFriction"].double()
    for g, name in enumerate(PAIRINGS):
      ids = L.pairIds[name]
      duration = getattr(self.sceneCfg.contacts, name).duration
      fields["pair_solref"][:, ids] = solrefBatch(p["restitution"][:, g], duration, self.sceneCfg.sim)[:, None]
      fields["pair_friction"][:, ids, :2] = p["contactFriction"][:, g, None, None].double()
    strength = p["motorStrength"][..., None].double()                         # (N, 2P, 1) over each car's drive actuators
    fields["actuator_gainprm"][:, L.driveAct, 0] *= strength
    fields["actuator_biasprm"][:, L.driveAct, 2] *= strength
    return fields

  def refreshGeometry(self) -> None:
    g = self.geometryTable[self.state["params", "variant"]]
    self.wheelbase, self.track, self.maxSteer = g[:, 0:1], g[:, 1:2], g[:, 2:3]
    self.worldSpawnHeight = g[:, 3]

  def close(self) -> None:
    self.backend.close()
