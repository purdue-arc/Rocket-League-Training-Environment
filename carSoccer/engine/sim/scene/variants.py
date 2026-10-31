import random, mujoco, numpy as np, carSoccer.config as cfg
from dataclasses import replace

STRUCTURE = ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "npair", "nsensordata")

# fields per-match randomisation writes, so every pack carries them even when a variant leaves them unchanged
RANDOMISED_FIELDS = ("body_mass", "body_inertia", "body_subtreemass", "geom_friction", "pair_solref", "pair_friction",
                     "actuator_gainprm", "actuator_biasprm")

def scaleCar(car: cfg.CarConfig, size: float, wheel: float, mass: float, motor: float) -> cfg.CarConfig:
  """One car variant. The wheel radius is clamped so the chassis still covers the wheels in plan view."""
  scaled = replace(
    car, length=car.length * size, width=car.width * size, height=car.height * size,
    wheelbase=car.wheelbase * size, track=car.track * size, axleDrop=car.axleDrop * size,
    mass=car.mass * mass, wheelMass=car.wheelMass * mass, knuckleMass=car.knuckleMass * mass,
    stallTorque=car.stallTorque * motor,
  )
  maxRadius = min(scaled.width - scaled.track, scaled.length - scaled.wheelbase) / 2 - 0.002
  return replace(scaled, wheelRadius=min(car.wheelRadius * wheel, maxRadius))

def sampleCars(car: cfg.CarConfig, variants: cfg.VariantConfig) -> list[cfg.CarConfig]:
  """variants.count car configs drawn once from variants.seed; index 0 is the nominal car."""
  rng = random.Random(variants.seed)
  draw = lambda bounds: rng.uniform(*bounds)
  return [car] + [scaleCar(car, draw(variants.sizeScale), draw(variants.wheelScale), draw(variants.massScale), draw(variants.motorScale))
                  for _ in range(variants.count - 1)]

def checkSameStructure(nominal: mujoco.MjModel, variant: mujoco.MjModel) -> None:
  diff = [f for f in STRUCTURE if getattr(nominal, f) != getattr(variant, f)]
  if diff:
    raise ValueError(f"car variant changes the model structure ({', '.join(diff)}); variants may only change parameters")

def arrayFields(model: mujoco.MjModel) -> dict[str, np.ndarray]:
  fields = {}
  for name in dir(model):
    if not name.startswith("_"):
      value = getattr(model, name, None)
      if isinstance(value, np.ndarray) and value.dtype.kind == "f":
        fields[name] = value
  return fields

def paramPacks(models: list[mujoco.MjModel]) -> list[dict[str, np.ndarray]]:
  """One parameter pack per model: every float field that differs between any two variants, plus the randomised fields.
  All packs share the same keys, so a world switches variant by copying one pack into its model."""
  nominal = arrayFields(models[0])
  others = [arrayFields(m) for m in models[1:]]
  keys = set(RANDOMISED_FIELDS) | {k for o in others for k, v in o.items() if not np.array_equal(v, nominal[k])}
  return [{k: fields[k].copy() for k in sorted(keys)} for fields in [nominal] + others]
