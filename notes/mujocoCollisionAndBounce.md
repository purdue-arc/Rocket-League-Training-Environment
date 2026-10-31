# How Objects Collide and Bounce in MuJoCo

Oct 30, 2026 · @pranav

In MuJoCo, objects never collide instantly. They overlap slightly, and a soft spring-damper constraint pushes them apart over several timesteps. How much speed survives that push is the bounce; there is no restitution setting.

## Overview

&#91;embedded content: MuJoCo contact pipeline · 6 stages per timestep\]

Steps 1 to 3 run during collision detection, 4 and 5 in the constraint solver, and 6 is what you observe. The sections below take each in turn.

## Which pairs can collide

Only geoms collide; bodies and joints never do. Every step, MuJoCo decides which geom pairs to test, then measures the distance between each candidate pair.

### Filtering

A pair is tested when all of these hold:

| Rule | Set with | Default |
| --- | --- | --- |
| Bitmask match: `(contype1 & conaffinity2) or (contype2 & conaffinity1)` is non-zero | `contype`, `conaffinity` on each geom | both 1, so everything collides with everything |
| Not in the same body | automatic | always excluded |
| Not parent and child body | `<option><flag filterparent="disable"/>` turns this off | excluded |
| Not listed in `<contact><exclude body1=... body2=.../>` | `<exclude>` | none |

Setting `contype="0" conaffinity="0"` makes a geom purely visual. Giving groups different bits lets you, say, make wheels touch the floor but not the ball.

### Explicit pairs

`<contact><pair geom1="..." geom2="..."/>` forces a pair to be tested regardless of the bitmasks, and carries its own contact parameters (`condim`, `friction`, `solref`, `solimp`, `margin`, `gap`). Those replace whatever the two geoms say. This is the cleanest way to give each surface pairing its own behaviour, for example ball-on-floor bouncing differently from ball-on-wall.

### Detection

Candidate pairs go through a cheap bounding-box test first, then an exact distance calculation. Primitive shapes (plane, sphere, capsule, box, cylinder, ellipsoid) use dedicated, fast routines. Meshes are treated as their convex hull and go through a general convex solver, which is slower and less smooth, so collision meshes are usually replaced by primitives.

The result is a signed distance per pair, plus a contact point and normal:

- distance > `margin`: no contact.
- distance < `margin`: a contact is created. A negative distance is penetration.
- The contact pushes only once distance < `margin` − `gap`. With both at their default of 0, contact starts exactly at touching.

MuJoCo has no instant collision. Objects overlap slightly and are pushed apart over several timesteps, so penetration is normal and expected.

## The soft contact model

Every contact is a constraint that behaves like a spring and a damper along the contact normal. Two parameter sets control it: `solref` (how stiff and how damped) and `solimp` (how much of that the solver actually enforces).

### The reference acceleration

For each contact, MuJoCo computes the acceleration it would like the constraint to have:

```latex
a_{ref} = -b \cdot v - k \cdot r
```

- `r`: penetration (distance minus margin), negative when overlapping.
- `v`: velocity along the normal.
- `k`: stiffness, pushes the objects apart more the deeper they overlap.
- `b`: damping, resists relative motion and removes energy.

`k` and `b` act on accelerations, not forces, so they are already divided by mass. The same `solref` gives the same contact behaviour for a light ball and a heavy one.

### solref: stiffness and damping

`solref` takes two numbers, in one of two forms:

| Form | Meaning | Converts to |
| --- | --- | --- |
| `solref="timeconst dampratio"` (positive) | time constant and damping ratio | `b = 2 / (dmax·timeconst)`, `k = d / (dmax²·timeconst²·dampratio²)` |
| `solref="-stiffness -damping"` (negative) | stiffness and damping directly | `b = damping / dmax`, `k = stiffness·d / dmax²` |

Default: `0.02 1`, a soft, critically damped contact that barely bounces.

In the positive form `k` depends on the damping ratio as well as the time constant, so lowering damping also stiffens the spring. The direct form avoids that coupling. With the `refsafe` flag on (the default), MuJoCo silently raises any positive time constant below 2 × timestep.

### solimp: impedance

`solimp="dmin dmax width midpoint power"` defines the impedance `d`, between 0 and 1. It ramps from `dmin` at zero penetration to `dmax` at penetration `width`, along a curve set by `midpoint` and `power`. Default: `0.9 0.95 0.001 0.5 2`.

`d` is how strongly the solver pursues `a_ref`. Near 1 the constraint is hard; lower values let it give way. Because the ramp's `width` is only 1 mm by default, `d` is effectively at `dmax` for any real impact, and `solimp` changes a bounce very little.

### The solver

All active constraints (contacts, joint limits, equality constraints) are solved together each step. The solver (`<option solver=...>`: Newton by default, or CG or PGS) finds forces that bring the actual accelerations close to `a_ref`, weighted by `d`. Contact forces may only push: a normal force that would pull is clamped to zero, and the contact lets go.

## When two geoms disagree

A contact has one set of parameters, but each of the two geoms has its own. MuJoCo picks or blends them in this order:

1. **Explicit `<pair>`**: its parameters are used, and the geoms' own are ignored.
2. **Different `priority`**: the geom with the higher priority wins outright, all parameters.
3. **Same priority**: blended as below.

| Parameter | Blending at equal priority |
| --- | --- |
| `condim` | the larger |
| `friction` | the larger, per component |
| `solref`, `solimp` | weighted average, weights from each geom's `solmix` (default 1, so a plain average) |
| `margin`, `gap` | the larger |

The averaging is the classic trap: setting a bouncy `solref` on a ball alone does nothing obvious when the floor keeps the soft default, because the two get averaged. Use `priority` for a single surface, and `<pair>` once several surfaces need different behaviour.

There is also a global override: `<option o_solref=... o_solimp=... o_friction=... o_margin=...>` with `<flag override="enable"/>` replaces these for every contact in the model. It is handy for experiments, not for a real scene.

## How bounce emerges

MuJoCo has no restitution parameter. Bounce is whatever speed survives the spring-damper of the previous sections, so you choose a bounce by choosing `k` and `b`.

### Two knobs: how much and how long

Write the contact as an oscillator, with `k = w0²` and `b = 2·zeta·w0`:

- **Damping ratio zeta** alone decides the coefficient of restitution e: the fraction of normal speed kept.
- **Natural frequency w0** decides how long the contact lasts. Half a damped oscillation takes `T = pi / (w0·sqrt(1 − zeta²))`.

So restitution and contact duration are independent choices. A stiffer contact is shorter and penetrates less; a softer one is longer and squashes deeper.

### The exact restitution formula

Because the contact can only push, it lets go when the spring-damper force reaches zero, slightly before the objects are back at the surface. Including that early release:

```latex
e = \exp\left(-\frac{\zeta}{\sqrt{1-\zeta^2}}\left(\pi - \arctan\frac{2\zeta\sqrt{1-\zeta^2}}{1-2\zeta^2}\right)\right)
```

It is valid for zeta up to 1/sqrt(2), so e from about 0.21 to 1. There is no closed-form inverse, but e falls steadily as zeta rises, so bisection finds zeta for a target e.

The often-quoted textbook formula, `zeta = −ln e / sqrt(pi² + (ln e)²)`, is the same model without the arctan term. It assumes the spring keeps acting until the objects separate, so it always predicts less bounce than MuJoCo produces.

### Setting a bounce

1. Pick e (measured on the real objects: drop height versus rebound height, `e = sqrt(h_out / h_in)`).
2. Pick a contact duration T. Real inflated balls stay in contact for roughly 5 to 15 ms.
3. Solve for zeta from e, then `w0 = pi / (T·sqrt(1 − zeta²))`.
4. Set `solref="-w0² -2·zeta·w0"` (direct form), on a `<pair>` or a higher-priority geom.

The direct form is preferred here: at e = 1, zeta is 0 and the positive form would need an infinite time constant.

### The timestep rule

The integrator steps through the contact in discrete jumps. If the contact lasts only a few timesteps, the spring is badly approximated and the bounce comes out wrong, usually too high, sometimes higher than the impact speed, which creates energy.

Keep the contact duration at **10 or more timesteps** (T / timestep ≥ 10). Below about 8 steps the error grows quickly. Note that the positive form hides this: a fixed time constant gives shorter contacts as the damping ratio drops, so very bouncy objects get the fewest steps.

Penetration is the price of a long contact. Peak overlap is roughly impact speed × T / pi, so a 10 ms contact at 15 m/s sinks about 5 cm into the surface. Resting overlap under gravity is tiny (about g / w0², under a millimetre).

## Friction and spin

The normal direction decides the bounce; the tangential directions decide sliding and spin. Each contact has its own frame: the normal plus two tangent axes.

### condim: which directions resist

| `condim` | Resists | Use for |
| --- | --- | --- |
| 1 | normal only, frictionless | ice, or cheap contacts that need no grip |
| 3 (default) | normal + sliding in both tangent directions | most contacts |
| 4 | + twisting about the normal (torsional) | objects spinning in place on a surface |
| 6 | + rolling about the tangent axes | balls that should slow down when rolling |

### friction: how much

`friction="sliding torsional rolling"`, default `1 0.005 0.0001`. Sliding is the ordinary Coulomb coefficient. Torsional and rolling only matter at `condim` 4 and 6 and have units of length (roughly the size of the contact patch). A `<pair>` takes five values, so the two tangent and two rolling directions can differ.

### Friction during a bounce

A ball arriving with sideways speed or spin is pushed by friction during the contact. The sideways impulse is at most the sliding coefficient times the normal impulse, however long the contact lasts. A sliding ball that hits the floor picks up spin; a spinning ball kicks sideways. This is how a bounce with spin changes direction.

There is no tangential restitution: friction can stop sliding at the contact point but never reverses it. Grippy, rubbery rebounds that real balls show are not modelled.

### Friction cone

`<option cone="pyramidal">` (default) approximates the friction limit with a pyramid; `"elliptic"` is exact and slightly slower. `impratio` (default 1) makes friction stiffer than the normal direction, which reduces slow creeping. Because contacts are soft, an object resting on a slope can creep very slowly even below the friction limit; `noslip_iterations` removes that if it matters.

## Reading contacts back

After `mj_step` (or `mj_forward`), every active contact is listed in `data.contact`, and `data.ncon` says how many there are.

| What | Where | Notes |
| --- | --- | --- |
| Number of contacts | `data.ncon` | 0 means nothing is touching |
| Which geoms | `data.contact.geom1`, `.geom2` | geom ids; `mujoco.mj_id2name` gives names |
| Penetration | `data.contact.dist` | negative when overlapping |
| Contact point | `data.contact.pos` | world coordinates, midway between the surfaces |
| Contact frame | `data.contact.frame` | 9 numbers; the first 3 are the normal |
| Parameters used | `data.contact.solref`, `.solimp`, `.friction`, `.dim` | after priority and mixing, useful for checking which won |
| Force | `mujoco.mj_contactForce(model, data, i, out)` | 6 numbers in the contact frame: normal force first, then friction |
| Velocity after a bounce | `data.qvel` of the free joint | linear part first, so vertical speed is index 2 |

For a total force on a body, MuJoCo's contact sensors (`<sensor><contact .../>`) sum contacts between chosen geoms or bodies without looping in Python.

A bounce is measured as rebound speed ÷ impact speed, both along the contact normal: the largest outward speed just after `ncon` returns to 0, divided by the inward speed just before contact.
