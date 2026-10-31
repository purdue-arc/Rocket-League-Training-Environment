import mujoco, numpy as np, carSoccer.config as cfg
from dataclasses import dataclass, field
from carSoccer.engine.sim.scene import arena, car, contacts, variants
from carSoccer.engine.sim.scene.layout import Layout, TEAMS, buildLayout, carNames

@dataclass
class Scene:
  model: mujoco.MjModel                      # nominal model (variant 0)
  layout: Layout                             # every index the upper layers need
  xml: str                                   # for the viewer and for MuJoCo Warp's speed test
  variants: list[dict[str, np.ndarray]] = field(default_factory=list)   # parameter pack per car variant; index 0 is nominal
  cars: list[cfg.CarConfig] = field(default_factory=list)                # the CarConfig each variant was built from

def addCarActuators(spec: mujoco.MjSpec, name: str, carCfg: cfg.CarConfig, geometry: car.CarGeometry) -> None:
  """Steering position servos on both front knuckles and a DC motor on every wheel."""
  lim = car.steerLimit(carCfg)
  for side in "LR":
    spec.add_actuator(
      name=f"{name}Steer{side}", target=f"{name}Steer{side}", trntype=mujoco.mjtTrn.mjTRN_JOINT,
      gear=[geometry.steerSign, 0, 0, 0, 0, 0],
      gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[carCfg.steerKp] + [0] * 9,
      biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -carCfg.steerKp] + [0] * 8,
      ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE, ctrlrange=[-lim, lim],
    )

  # DC motor: torque = stall * (u - w / wFree), clipped to +-stall
  wFree = carCfg.topSpeed / geometry.wheelRadius
  for wheel in car.WHEELS:
    spec.add_actuator(
      name=f"{name}Drive{wheel}", target=f"{name}Drive{wheel}", trntype=mujoco.mjtTrn.mjTRN_JOINT,
      gear=[geometry.driveSign, 0, 0, 0, 0, 0],
      gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[carCfg.stallTorque] + [0] * 9,
      biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, 0, -carCfg.stallTorque / wFree] + [0] * 7,
      ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE, ctrlrange=[-1, 1],
      forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE, forcerange=[-carCfg.stallTorque, carCfg.stallTorque],
    )

def compileScene(sceneCfg: cfg.SceneConfig, carCfg: cfg.CarConfig):
  """Arena, ball and 2P copies of one car. Returns (spec, model, pairing name -> pair names, probed car geometry)."""
  P = sceneCfg.carsPerTeam
  geometry = car.probeCar(carCfg)
  spec = mujoco.MjSpec.from_string(arena.arenaXML(sceneCfg.arena, sceneCfg.ball, sceneCfg.sim))

  names = carNames(P)
  for k, name in enumerate(names):
    team, slot = TEAMS[k // P], k % P
    sign = 1 if team == "blue" else -1      # blue starts on -x facing +x; the kickoff mutator places cars properly
    frame = spec.worldbody.add_frame(
      pos=[-sign * 2.0, (slot - (P - 1) / 2) * 1.5, geometry.spawnHeight],
      quat=[1, 0, 0, 0] if team == "blue" else [0, 0, 0, 1],
    )
    body = frame.attach_body(car.carSpec(carCfg).body("Chassis"), name, "")
    body.add_freejoint(name=name)
    spec.geom(f"{name}Chassis").material = "carBlue" if team == "blue" else "carOrange"
    addCarActuators(spec, name, carCfg, geometry)
    spec.add_sensor(
      name=f"{name}Touch", type=mujoco.mjtSensor.mjSENS_CONTACT,
      objtype=mujoco.mjtObj.mjOBJ_GEOM, objname="ball", reftype=mujoco.mjtObj.mjOBJ_GEOM, refname=f"{name}Chassis",
      intprm=[1, 1, 1],                      # data = found, reduce = mindist, one contact
    )

  wallGeoms = [piece[0] for piece in arena.wallGeoms(sceneCfg.arena)]
  pairNames = contacts.addPairs(spec, sceneCfg.contacts, sceneCfg.sim, wallGeoms, [f"{n}Chassis" for n in names])
  return spec, spec.compile(), pairNames, geometry

def buildScene(sceneCfg: cfg.SceneConfig) -> Scene:
  """Config -> one compiled MjModel plus its Layout and car variants. Built once, on CPU, before training.
  Each variant compiles the whole scene with a different car, is checked to have the same structure, and keeps only
  a parameter pack of the fields that differ."""
  cars = variants.sampleCars(sceneCfg.car, sceneCfg.variants)
  spec, model, pairNames, geometry = compileScene(sceneCfg, cars[0])
  models, geometries = [model], [geometry]
  for carCfg in cars[1:]:
    _, variantModel, _, variantGeometry = compileScene(sceneCfg, carCfg)
    variants.checkSameStructure(model, variantModel)
    models.append(variantModel)
    geometries.append(variantGeometry)

  layout = buildLayout(model, sceneCfg.carsPerTeam, pairNames, geometries)
  return Scene(model=model, layout=layout, xml=spec.to_xml(), variants=variants.paramPacks(models), cars=cars)
