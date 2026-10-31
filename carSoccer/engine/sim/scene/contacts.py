import itertools, math, mujoco, torch, carSoccer.config as cfg
from torch import Tensor
from carSoccer.engine.sim.scene import ball

PAIRINGS = ("ballFloor", "ballWall", "ballCar", "carWall", "carCar")   # order of per-pairing params everywhere

def pairParams(contact: cfg.ContactConfig, sim: cfg.SimConfig) -> dict:
  """add_pair keyword arguments for one pairing; contactSolref enforces the minimum contact steps."""
  e = max(contact.restitution, ball.E_MIN)  # car pairings below E_MIN are clamped (guide Section 5, open decision)
  f = contact.friction

  return dict(
    condim=contact.condim,
    friction=[f, f, contact.torsional, contact.rolling, contact.rolling],
    solref=list(ball.contactSolref(e, sim, contact.duration)),
  )

def addPairs(spec: mujoco.MjSpec, contacts: cfg.ContactsConfig, sim: cfg.SimConfig,
             wallGeoms: list[str], chassisGeoms: list[str]) -> dict[str, list[str]]:
  """Add one <pair> per colliding geom pair, grouped by pairing. Returns pairing name -> pair names, in the order added."""
  groups = {                               # keys in PAIRINGS order
    "ballFloor": (contacts.ballFloor, [("ball", "floor")]),
    "ballWall": (contacts.ballWall, [("ball", w) for w in wallGeoms]),
    "ballCar": (contacts.ballCar, [("ball", c) for c in chassisGeoms]),
    "carWall": (contacts.carWall, [(c, w) for c in chassisGeoms for w in wallGeoms + ["floor"]]),
    "carCar": (contacts.carCar, list(itertools.combinations(chassisGeoms, 2))),
  }
  names = {}
  for group, (contact, geoms) in groups.items():
    params = pairParams(contact, sim)
    names[group] = []
    for g1, g2 in geoms:
      name = f"{group}:{g1}:{g2}"
      spec.add_pair(name=name, geomname1=g1, geomname2=g2, **params)
      names[group].append(name)

  return names

def zetaTable(points: int = 4097) -> tuple[Tensor, Tensor]:
  """(restitution ascending, damping ratio) sampled over 0 <= zeta <= 1/sqrt(2), for interpolating ball.dampingRatio."""
  zeta = torch.linspace(0, 1 / math.sqrt(2), points, dtype=torch.float64)
  e = torch.tensor([ball.restitution(z) for z in zeta.tolist()], dtype=torch.float64)
  return e.flip(0), zeta.flip(0)

E_GRID, ZETA_GRID = zetaTable()

def solrefBatch(e: Tensor, duration: float, sim: cfg.SimConfig) -> Tensor:
  """Batched contactSolref: restitution e (...) -> direct-form solref (..., 2), float64. Used to randomise a pairing's
  bounce per world; the damping ratio is interpolated from a table of ball.restitution instead of bisected."""
  if duration / sim.timestep < sim.minContactSteps:
    raise ValueError(f"contact lasts {duration / sim.timestep:.1f} timesteps, need at least {sim.minContactSteps}")
  e = e.double().clamp(ball.E_MIN, 1.0)
  i = torch.searchsorted(E_GRID, e).clamp(1, E_GRID.shape[0] - 1)
  e0, e1, z0, z1 = E_GRID[i - 1], E_GRID[i], ZETA_GRID[i - 1], ZETA_GRID[i]
  zeta = z0 + (z1 - z0) * (e - e0) / (e1 - e0)
  w0 = math.pi / (duration * torch.sqrt(1 - zeta ** 2))
  return torch.stack([-w0 ** 2, -2 * zeta * w0], -1)
