# Car Soccer Dev Guide

Oct 27, 2026 · @pranav

## How to use this doc

This guide explains the car-soccer simulator built so far, why each piece exists, and how to rebuild it by hand as a cleaner v2. That v2 runs fast on CPU MuJoCo today and can move to a GPU later without rewriting the layers above the simulator.

Read it in this order:

1. Sections 1 to 3: what the project is, what v1 contains, and what we learned (with measured numbers).
2. Sections 4 and 4b: the v2 architecture, its rules, and the component and paradigm API you use daily. Everything after depends on them.
3. Sections 5 to 11: one section per layer, each with its contract (inputs, outputs, shapes) and a sketch of the code.
4. Section 13: the milestone plan you follow while coding, and Section 15 before touching the real cars.

### Standing instructions for the AI assistant

Paste this doc into your assistant's context, then this block as its instructions:

```text
You are a tutor helping me build the car-soccer project described in this guide.
- Teach, don't do. Explain the concept, point to the guide section, and let me write the code.
  Write full functions only when I explicitly ask; otherwise give signatures, pseudocode or a hint.
- Hold me to the contracts in the guide: tensor shapes, units, layer boundaries.
  If my code breaks a contract or a speed rule (Section 4), say so and cite the rule.
- Before I move on from a milestone, ask me to run its done-when test (Section 13).
- When I ask why, explain the physics or RL reasoning, not just the code.
- If the guide and my request conflict, point out the conflict and ask which wins.
- If something in the guide looks wrong or outdated, say so; don't silently follow it.
```

Conventions used throughout: SI units (metres, kilograms, seconds, radians); `N` = number of parallel matches (worlds); `P` = cars per team (3); shapes are written like `(N, 2P, 3)`.

## 1. Project overview

The goal is a policy that plays 3v3 car soccer on real robot cars, trained entirely in simulation. The simulator exists only to train and test that model; the real cars are controlled separately and run the exported network.

Two requirements shape everything:

- **Car-model invariance.** The policy must work on any car within a range of sizes, masses and motors, so training randomises the car itself.
- **Speed.** Target 20k+ environment steps per second on CPU MuJoCo now, with a path to GPU (MuJoCo Warp) later.

### Real setup in simulator units

| Item | As specified | In the sim | Status |
| --- | --- | --- | --- |
| Field | 30 × 20 ft | 9.144 × 6.096 m | Given |
| Ball | 8 in radius | r = 0.2032 m, 0.45 kg hollow | Mass assumed |
| Goals | none given | 1.50 m wide × 0.60 m tall × 0.50 m deep | Assumed |
| Walls | none given | 0.50 m high | Assumed |
| Car | a bit bigger than RC | 0.46 m long, \~0.31 m wide, 3.5 kg, 4WD, Ackermann steering | Assumed |
| Top speed | 50 mph | 22.35 m/s | Given |
| Usual speed | 10–20 mph | 4.5–8.9 m/s | Given |
| Control loop | not given | 50 Hz, policy decides every 3rd tick (\~17 Hz) | Assumed |

Every assumed value lives in one config file. Replace it with a measurement of your hardware when you have one.

### Terms used in this guide

| Term | Meaning |
| --- | --- |
| World | One independent match. The simulator steps `N` worlds in parallel. |
| Env step | One control tick (20 ms at 50 Hz) of every world. Speed is measured in env steps per second summed over worlds. |
| Substep | One physics step. An env step contains `decimation` substeps (20 at 1 ms, 10 at 2 ms). |
| Decision | One policy forward pass. With `action_repeat = 3`, one decision covers 3 env steps. |
| Team frame | Coordinates rotated so the team always attacks toward +x. Orange's view is rotated 180°. |
| Agent | Whatever one policy call controls: a whole team, or one car. |
| Backend | The code that advances physics: CPU MuJoCo now, MuJoCo Warp later. |

## 2. What exists today (v1)

v1 is a working, tested prototype: 16/16 physics checks pass and a 2M-step training run beats an untrained policy by 13 goals per match. Its structure is what v2 fixes: numpy and per-world Python loops are baked into the env, and rewards, models and training are hard-wired together.

### Repo tour (`car_soccer_mujoco/`)

| File | What it does | Keep in v2? |
| --- | --- | --- |
| `car_soccer/config.py` | Dataclasses for field, car, ball, sim, game, randomisation, car variants, rewards. `cfg.nominal()` turns all noise off. | Keep the values; split into per-layer configs |
| `car_soccer/model_builder.py` | Builds the MuJoCo model with `MjSpec`: arena MJCF, ball with per-surface contact pairs, cars attached as child specs (rectangular car or URDF). Measures wheelbase, track, wheel radius and axis signs into a `CarLayout`. | Keep almost as is (Scene layer) |
| `car_soccer/env.py` | The batched env: actuation, physics stepping, observations, rewards, goals, kickoffs, flip recovery, randomisation, car variants. | Split into Backend, Game, Agents, Rewards |
| `car_soccer/scripted.py` | Rule-based team: attacker, supporter, defender, plus unstick behaviour. | Keep; port to torch |
| `car_soccer/render.py` | Offscreen mp4 recording of one world. | Keep |
| `car_soccer/rl/features.py` | Converts the 87 raw numbers into per-car ego features (ball 1.2 m ahead, 20° left...). | Becomes an observation view |
| `car_soccer/rl/policies.py` | One `TeamPolicy` class with three hard-coded paradigms. | Replace with the model pipeline |
| `car_soccer/rl/ppo.py` | PPO trainer, curriculum stages, opponents, logging, evaluation, all in one file. | Replace with the training framework |
| `scripts/physics_check.py` | 16 physics and game-rule tests. | Keep; becomes the test suite |
| `scripts/inspect_car.py` | Detects URDF joint roles, measures the car, renders 4 views. | Keep |
| `scripts/train.py`, `evaluate.py`, `plot_training.py`, `record_match.py` | CLI tools. | Rewrite around experiment files |
| `docs/CAD_TO_URDF.md`, `docs/TRAINING.md` | Export guide and training guide. | Keep, update |

### One env step in v1, in order

1. Split the two `(N, 6)` team actions into `(N, 6 cars, 2)` and clip to \[−1, 1\].
2. Delay them by the per-world latency (1–3 steps), add command noise.
3. Steering: slew-limit toward `steer × max_steer + trim`, then split into left and right wheel angles with Ackermann geometry.
4. Throttle goes straight to each driven wheel's DC-motor actuator.
5. Compute the ball's air drag and Magnus force, then run `decimation` physics substeps for every world (a Python thread pool).
6. Copy every world's `qpos`, `qvel` into numpy (a per-world Python loop).
7. Check goals, compute reward terms, restart play after a goal or a ball out of the field, right flipped cars.
8. Build both teams' 87-number observations with sensor noise and dropped frames; auto-reset worlds whose match ended.

### The contract v1 exposes

- **Action, per team:** `(N, 6)` = `[t0, s0, t1, s1, t2, s2]`, throttle and steer per car, each in \[−1, 1\].
- **Observation, per team:** `(N, 87)` float32 in that team's frame, raw SI units:
  - 6 cars × 13 (own team first): `x y z roll pitch yaw vx vy vz wx wy wz steer`
  - ball × 6: `x y z vx vy vz`
  - game × 3: `own_score opp_score time_remaining`
- **Step returns:** `obs, rew, terminated, truncated, info`; `info` holds `reward_terms`, `goals`, `score`, and `final_obs` / `final_score` for worlds that just ended.

v2 keeps this contract as one adapter (useful for deployment and comparison), but the core moves to structured state, described from Section 4 on.

## 3. Lessons learned

Each item below cost real debugging time in v1; rebuild with them in mind rather than rediscovering them.

### Physics and modelling

| Lesson | Evidence | What to do |
| --- | --- | --- |
| MuJoCo has no restitution parameter; bounce comes from contact damping. | Theory alone gave the wrong bounce; a 1.25 factor fixed it. | Use the calibrated formula below, then verify with a drop test. |
| Small, fast balls need a small timestep. | A 3 in ball at 1 ms bounced with e > 1.5 at 10 m/s. The 16 in ball is fine at 1 ms (e = 0.64 / 0.62 / 0.59 at 3 / 8 / 15 m/s) and close at 2 ms (0.64–0.68). | Auto timestep: 0.5 ms below 6 cm radius, else 1 ms; test 2 ms for speed. |
| MuJoCo's built-in air model applies drag to every body. | Step cost rose from 54 to 98 µs. | Apply drag + Magnus to the ball only, via `xfrc_applied`. |
| The inner front wheel turns further than the centre steer angle. | Joint limits at `max_steer` stalled sharp turns. | Limit = Ackermann inner angle at full lock + 0.05 rad. |
| Collision shapes must not overlap inside one car. | A chassis box wider than the track touched the wheel spheres; turning stalled. | Narrow the chassis collider; exclude intra-car contacts. |
| Sphere wheels roll smoothly; mesh wheels don't. | Faceted CAD wheels gave 1.67 vs 1.86 m/s at the same throttle. | Replace wheel collision with a sphere of the measured radius. |
| A ball taller than the cars is pinned easily. | 16 in ball vs \~0.12 m cars: play often stalls against a wall. | Open question: confirm 8 in radius vs diameter. |

Restitution e to MuJoCo damping ratio (with contact time constant = 8 × timestep, min 4 ms):

```latex
\zeta = 1.25 \cdot \frac{-\ln e}{\sqrt{\pi^2 + (\ln e)^2}}
```

### Reinforcement learning

| Lesson | Evidence | What to do |
| --- | --- | --- |
| Exploring at 50 Hz cancels itself out. | No learning in 250k steps; random 50 Hz commands average to almost no motion. | Hold each decision for 3 ticks (`action_repeat = 3`). |
| Raw world coordinates are hard to learn from. | Same reward, no progress with the 87 raw numbers. | Give the network car-relative features (ball bearing, distance). |
| Standard kickoffs waste data. | The car reaches the ball in \~1 s, then pins it for the rest of the match. | Randomise kickoffs (100% early, 25% late). |
| Reward mix matters early. | Approach speed after 250k steps: 0.10 m/s with the full mix vs 0.20 approach-first vs 0.26 with short random matches; 0.23 with full noise. | Schedule weights by stage: approach first, goals later. |
| Rare large rewards swamp the critic. | Goal reward of 10 vs per-step shaping of \~0.01. | Normalise rewards by a running std of the return. |
| PPO itself was fine. | A trivial reward (mean throttle) learned in 15 iterations. | When learning stalls, test the trainer on a trivial reward first. |

### Results to beat

2M steps (`shared_car`, 45 min on 2 cores, full noise and car variants) vs scripted bots, 16 × 60 s matches: win / draw / loss 6% / 81% / 12%, goals 0.25 for / 0.31 against, ball in the opponent half 67% of the time. An untrained policy concedes 13.4 goals per match. Scoring reliably needs the self-play stage and tens of millions of steps.

## 4. v2 architecture

v2 uses the same split as RLGym and GigaLearnCPP: an **engine** that only knows how the world moves, a set of small **components** that define the game for the learner, and a **learner** on top. The engine is the only part that differs between MuJoCo on CPU, MuJoCo Warp on GPU and the real cars; every component runs unchanged on all three. Unlike RLGym (a Python dict per agent) and GigaLearnCPP (C++ code per game), every component here works on the whole batch of worlds at once as PyTorch tensors, which is where pure Python gets its speed.

| Part | Job | In simulation | On the real cars |
| --- | --- | --- | --- |
| Engine | Advance one control tick; report state and events; place cars for a reset | `SimEngine` = Scene + Backend + car and ball physics (Sections 5–7) | `RealEngine` = tracking + radio + positioning (Section 15) |
| StateMutator | Choose the next kickoff or reset state, and per-match physics randomisation | Teleports cars and ball | Targets the cars drive themselves to; an operator places the ball |
| ObsBuilder | State → each agent's observation | Section 8 | Same code |
| ActionParser | Policy output → throttle and steer per car | Section 8 | Same code |
| RewardFunction | State + events → reward per agent | Section 9 | Logged only |
| DoneCondition | Match over: terminated or truncated | Section 7 | Same code |
| Paradigm | Named bundle: agent scheme + ObsBuilder + ActionParser + model | Section 4b | Same bundle |
| Learner | Models, PPO, league, curriculum, runner | Sections 10–11 | Only the exported actor runs |

&#91;embedded content: one decision through the seven layers\]

During one decision, actions flow down the left column and state flows back up. The Scene is built once before training. Only the highlighted parts change: the engine is `SimEngine` (with the CPU or GPU backend) in training and `RealEngine` on the cars, which needs no backend.

### Why PyTorch above the backend

- **CPU now:** `mujoco.rollout` writes into numpy arrays; `torch.from_numpy` wraps them with no copy.
- **GPU later:** MuJoCo Warp arrays convert with `wp.to_torch`, also with no copy.
- **Models and training** are PyTorch anyway, so observations never change format.
- **Cost:** a torch op on CPU has a few microseconds of fixed overhead, so keep batches large (`N` ≥ 256) and fuse work with `torch.compile` once the code is stable.
- **Structured data:** use `tensordict.TensorDict` for observations, rollouts and state; it batches, slices, moves devices and stacks like one tensor.

### The rules every module follows

These are what keep v2 fast on CPU and portable to GPU. The tutor should flag any code that breaks one.

1. **No per-world Python loops in the step path.** Select worlds with boolean masks (`done`, `scored`) and write with `torch.where` or masked indexing.
2. **One device, no hidden syncs.** Nothing in `step()` calls `.item()`, `.cpu()`, `.numpy()`, `print` on a tensor, or branches on a tensor value. Each forces a GPU sync.
3. **Fixed shapes for the whole run.** `N`, car count, observation and action sizes never change. Variable things (frozen opponents, cars out of play) are masks.
4. **Pure functions where possible.** Game rules, observations and reward terms are functions `state → tensor` with no hidden state, so they are easy to test and to compile.
5. **Structure fixed, parameters random.** The MuJoCo model is built once; randomisation changes per-world parameters (mass, friction, gains), never the joint or geom count.
6. **Resets are masked writes inside `step()`** (auto-reset), never a separate per-world call.
7. **Randomness from one seeded `torch.Generator`** on the env's device.
8. **Config is data, behaviour is registered.** Dataclasses hold numbers; reward terms, models and opponents are looked up by name in registries.
9. **One clock.** Schedules read a single global progress value (env steps so far); nothing keeps its own counter.
10. **Every layer ships with a test** that runs in seconds.

Rule 11, **TensorDict at the boundaries only:** pass TensorDicts between components; inside a hot function, pull out plain tensors once and work on those. No nested TensorDict operations inside loops, and flatten before anything goes into `torch.compile`.

### v2 repo layout

```text
car_soccer/
  engine/
    base.py          Engine protocol, GameState, StepEvents
    sim/             scene/ (build, cars, variants, layout)  backend/ (cpu_rollout, warp)
                     physics.py (actuation, aero)  sensors.py  rules.py  sim_engine.py
    real/            tracking.py  radio.py  safety.py  positioning.py  real_engine.py
  components/
    mutators.py      kickoffs, scattered starts, physics randomisation, car variants
    obs/             frames.py  terms.py  builders.py
    actions.py       continuous, binned lookup table, team vs car
    rewards/         api.py  terms.py  schedules.py
    done.py          match clock, no-touch timeout
    estimator.py     Kalman filter: raw measurements -> estimated state (sim and real)
  env.py             SoccerEnv = Engine + components + roster + action repeat
  paradigms.py       named bundles: team, car_shared, car_heads, car_discrete
  models/            registry.py  encoders.py  cores.py  heads.py  pipeline.py
  learn/             ppo.py  collector.py  league.py  skill.py  curriculum.py  runner.py  report.py
  deploy/            export.py  policy_runner.py
experiments/         base.py  quick.py  league_transformer.py
scripts/             train.py  evaluate.py  bench.py  inspect_car.py  record.py  sysid.py
tests/
```

## 4b. Components and paradigms: the API you use every day

Day to day you touch three things: components, paradigms and the experiment file. Everything else is built once and left alone.

### Component interfaces (batched versions of RLGym's)

```python
class StateMutator:
    def apply(self, state: GameState, mask: Tensor, ctx: Ctx) -> None:
        """Write the next start state for worlds where mask is True (in place)."""

class ObsBuilder:
    spec: ObsSpec                                      # entity names and feature sizes
    def reset(self, mask: Tensor) -> None: ...         # clear per-world history
    def build(self, state: GameState, ctx: Ctx) -> TensorDict: ...   # (B_agents, ...)

class ActionParser:
    space: ActionSpace                                 # Box(2) / Box(6) / Discrete(35) ...
    def parse(self, raw: Tensor, ctx: Ctx) -> Tensor: ...  # -> car_actions (N, 2P, 2)

class DoneCondition:
    kind: Literal["terminated", "truncated"]
    def check(self, state: GameState, events: StepEvents) -> Tensor: ...  # (N,) bool

# RewardFunction is the expression API of Section 9.
```

`ctx` is the batched equivalent of RLGym's `shared_info` and GigaLearnCPP's per-player data: quantities computed once per step (team-frame state, each car's distance and bearing to the ball, `prev` state, the progress clock) that every component reads instead of recomputing.

### How `SoccerEnv.step` calls them

1. `ActionParser.parse` turns every agent group's raw output into `car_actions`.
2. The engine steps `action_repeat` times; events are OR-ed across the ticks.
3. `ctx` is rebuilt from the new state.
4. `RewardFunction` and the `DoneCondition`s run.
5. For worlds that are done or need a kickoff, the `StateMutator`s write a start state and the engine applies it.
6. `ObsBuilder.build` produces the next observations, and the final observations of finished matches.

### Paradigms: one line to switch

A paradigm is a named bundle that fixes everything that must agree with each other: the agent scheme, the ObsBuilder, the ActionParser, and a default model.

| Paradigm | Agent | Observation | Actions | Default model |
| --- | --- | --- | --- | --- |
| `team` | one per team | entity view, team frame | continuous, 6 | `flat_mlp` |
| `car_shared` | one per car | entity view, own frame | continuous, 2 | `flat_mlp`, shared weights |
| `car_heads` | one per car | same + slot index | continuous, 2 | shared encoder, one head per slot |
| `car_discrete` | one per car | same | 35-entry lookup table (5 throttle × 7 steer) | `flat_mlp` + categorical head |

```python
experiment = base.replace(paradigm=paradigms.car_shared(model="entity_transformer"))
```

or from the shell: `python scripts/train.py experiments/base.py --paradigm team --model gru_mlp`. Rewards, mutators, done conditions, engine and learner settings stay as they are. A checkpoint belongs to the paradigm it was trained in; moving a skill to a new paradigm means distillation (train the new policy to imitate the old one), not loading weights.

### End-of-day training

The runner mirrors GigaLearnCPP's `LearnerConfig`: one object holds the run, and a step callback gives you a metrics report.

```python
run = Learner(experiment, LearnerConfig(
    num_worlds=1024, steps_per_iteration=65_536,
    ppo=PPOConfig(policy=dict(layers=(256, 256)), critic=dict(layers=(512, 512))),
    standardize_obs=True, standardize_returns=True,
    skill_tracker=SkillTrackerConfig(every=25, matches=32),   # Elo against saved policy versions
    save_every=25, device="cpu", backend="cpu_rollout",
))
run.on_iteration(lambda report, state: report.add_avg("ball_speed", state.ball.vel.norm(dim=-1)))
run.learn()
```

Every reward term, PPO diagnostic, throughput number and Elo rating lands in the same report, written to CSV and optionally TensorBoard or Weights & Biases.

## 5. Layer 1: Scene

The Scene layer turns config into one compiled `MjModel` plus a `Layout`, once, on CPU, before training starts. Both backends consume the same model, so this layer never changes for the GPU move.

### Contract

```python
def build_scene(cfg: SceneConfig) -> Scene: ...

@dataclass
class Scene:
    model: mujoco.MjModel        # nominal model (variant 0)
    layout: Layout               # every index the upper layers need
    variants: list[ParamPack]    # per-variant parameter overrides, same structure
    xml: str                     # for the viewer and for MuJoCo Warp's speed test
```

`Layout` is a frozen dataclass of integer index tensors, so upper layers never look anything up by name inside the step:

| Field | Shape | Meaning |
| --- | --- | --- |
| `car_qpos` | `(2P,)` | qpos address of each car's free joint (pos 3 + quat 4) |
| `car_qvel` | `(2P,)` | qvel address (lin 3 + ang 3, angular in body frame) |
| `steer_qpos` | `(2P, 2)` | left and right steering joint angles |
| `steer_act`, `drive_act` | `(2P, 2)`, `(2P, D)` | actuator ids; `D` = driven wheels |
| `ball_qpos`, `ball_qvel`, `ball_body` | scalars | ball addresses |
| `car_bodies` | list of `2P` id lists | for mass randomisation |
| `geometry` | per variant | wheelbase, track, wheel radius, spawn height, max steer, axis signs |

### How the model is built (same as v1)

1. **World MJCF string:** floor, walls, corner chamfers, goals, lights, cameras, ball. Physics options: `integrator="implicitfast"`, timestep from config.
2. **Ball:** an explicit `<inertial>` (hollow shell: 2/3 m r²), and `<pair>` contacts so floor and walls each get their own bounce (Section 3 formula) and rolling resistance (`friction` slot 4 = Crr × r).
3. **Cars:** each car is a child `MjSpec` (rectangular car or URDF), attached under a body with a free joint named `blue0`, `orange2`, … with every name prefixed `blue0_`.
4. **Probe:** compile one car alone at the origin and measure wheel positions and joint axes. This catches rotated CAD frames, millimetre units and mirrored axles.
5. **Actuators,** added with the spec API:
   - Steering: position servo per front knuckle, `gear` = the measured axis sign.
   - Drive: one per driven wheel, a DC-motor model `τ = stall · (u − ω / ω_free)`, clipped to ±stall. In MuJoCo terms: `gaintype=fixed`, `gainprm[0] = stall`, `biastype=affine`, `biasprm[2] = −stall / ω_free`, where `ω_free = top_speed / wheel_radius`.
6. **Contact filtering:** exclude every pair of bodies within one car.

### Collision groups (new in v2, for speed)

Give each geom a `contype` / `conaffinity` bit so MuJoCo never tests pairs that can't matter:

| Group | Collides with |
| --- | --- |
| Floor | wheels, ball, chassis (flipped cars) |
| Walls and goals | chassis, ball, and wheels only if they stick out past the bumpers (in v1 they do, by 1.5 cm a side) |
| Chassis | floor, walls, ball, other cars' chassis |
| Wheels | floor (plus walls, per the row above) |
| Ball | floor, walls, chassis |

Also set `cone="pyramidal"` and cap `iterations` / `ls_iterations`; measure the bounce test before and after each change.

**Tyre grip is the weakest part of the car model.** Sphere wheels with Coulomb friction give the same grip in every direction and a sharp switch from grip to slide; real tyres slide gradually (slip angle), have less lateral than longitudinal grip, and behave differently on turf than on a gym floor. A pyramidal cone also makes grip depend slightly on direction, so compare it against the elliptic cone in the turning tests before keeping it. Fit the floor friction from real drives (Section 15) and randomise around the fit. If cornering still doesn't match, the next step is a custom tyre force (a slip-angle curve applied through `xfrc_applied` at each wheel) instead of contact friction.

### Car variants as parameter packs (changed in v2)

v1 compiled one model per car variant and swapped whole models between worlds. v2 compiles each variant once to check that its structure matches, then keeps only a **parameter pack**: the model fields that differ (`body_pos`, `body_mass`, `body_inertia`, `geom_size`, `geom_rbound`, `geom_aabb`, actuator gains and limits). Applying a variant to a world means copying those fields into that world's parameters. On CPU each world owns a model copy; on GPU those fields become batched per world. The upper layers only see `geometry[variant_id]` tensors indexed by world.

## 6. Layer 2: Simulation backend

The backend is the only code that touches MuJoCo during training, and the only layer that differs between CPU and GPU. It advances `N` worlds by `nsub` substeps per call and exposes their state as tensors.

### Contract

```python
class SimBackend(Protocol):
    num_worlds: int
    device: torch.device
    qpos: Tensor          # (N, nq)  live views, updated by step()
    qvel: Tensor          # (N, nv)
    sensordata: Tensor    # (N, nsensor), e.g. ball-contact flags

    def step(self, ctrl: Tensor, xfrc: Tensor | None, nsub: int) -> None:
        """ctrl (N, nu) and xfrc (N, nbody, 6) are held constant for nsub substeps."""
    def write_state(self, mask: Tensor, qpos: Tensor, qvel: Tensor) -> None:
        """Overwrite worlds where mask is True (resets, kickoffs, flip recovery)."""
    def set_params(self, mask: Tensor, params: dict[str, Tensor]) -> None:
        """Per-world model fields (friction, mass, gains, a variant's ParamPack)."""
```

Everything above this layer reads `qpos`, `qvel` and `sensordata` and never imports `mujoco`.

### CPU backend with `mujoco.rollout`

v1 called `mj_step` from Python 20 times per world per env step. At 20k env steps per second that is 400k Python calls a second, which the Python interpreter lock can't sustain. `mujoco.rollout` runs every world's substeps inside C++ threads instead:

- **One call per env step** for all worlds: `nstep = nsub`, a persistent thread pool (`mujoco.rollout.Rollout(nthread=...)`), one `MjData` per thread, and a list of `N` models (one per world, so randomised parameters work).
- **State in one array:** keep `state_np` shaped `(N, nstate)` using `mjSTATE_FULLPHYSICS`. It is the rollout's `initial_state`; after the call, copy the last substep back into it. `qpos` and `qvel` are `torch.from_numpy` views into slices of it, so there is no copy into torch.
- **Controls:** pass `(N, 1, ncontrol)` (the single step is tiled across substeps) with `control_spec = mjSTATE_CTRL | mjSTATE_XFRC_APPLIED`, so the ball's drag force rides along with the actuator commands.
- **Preallocate** the output buffer `(N, nsub, nstate)` once and pass it in every call.
- **Resets** are masked writes into `state_np`; the next rollout starts from them, so no `mj_forward` is needed.
- **Sensors** (for example a contact sensor between each car and the ball) come back in the `sensordata` output.
- **Tuning:** `nthread` = cores minus 2 for PyTorch; try `chunk_size`; keep `N` between 256 and 1024.
- **Known cost:** the solver's warm start isn't carried between calls, so each env step starts the solver cold. Measure whether passing `initial_warmstart` helps.

### GPU backend with MuJoCo Warp (later)

The same contract on an NVIDIA GPU:

- Convert once: a Warp model from the same `MjModel`, and Warp data with `nworld = N`.
- `step()` runs the Warp step `nsub` times, ideally captured once as a CUDA graph.
- `qpos`, `qvel`, `sensordata` are `wp.to_torch(...)` views; `write_state` and `set_params` are masked tensor writes.
- Randomisation uses per-world model fields. Check which fields your MuJoCo Warp version can batch per world before relying on car-geometry variants there.

Function and argument names here are as of late 2026; check the MuJoCo Warp docs for your installed version.

### Parity test (required before trusting a new backend)

Run both backends from the same initial state with the same controls for 1 s of game time and compare: car and ball positions within a few millimetres, and the same goal events. The physics checks from v1 must pass on each backend.

## 7. Layer 3: The engine (game physics, rules, sensors)

The engine turns per-car commands into a new world state: it applies the car's real-world imperfections, steps physics, detects goals and touches, and reports both the true state and what the sensors would measure. `SimEngine` wraps Scene + Backend; `RealEngine` (Section 15) implements the same contract with tracking and radios. Neither knows about policies, observations or rewards.

### Contract

```python
class Engine(Protocol):
    num_worlds: int                      # SimEngine: N; RealEngine: 1
    device: torch.device
    state: GameState                     # true state (sim) + .measured (what sensors report)
    capabilities: EngineCaps             # teleport, true_state, parallel worlds

    def step(self, car_actions: Tensor) -> StepEvents:
        """car_actions: (N, 2P, 2) = [throttle, steer] per car, blue first, in [-1, 1]. One control tick."""
    def set_state(self, mask: Tensor, target: GameState) -> None:
        """Sim: teleport masked worlds. Real: drive cars to the targets, wait for the ball, then return."""
    def set_params(self, mask: Tensor, params: dict[str, Tensor]) -> None:
        """Sim: per-world physics parameters. Real: ignored."""
```

Taking per-car actions (not per-team) keeps this layer free of any notion of who controls which car.

### `GameState` (a TensorDict, world frame)

| Field | Shape | Notes |
| --- | --- | --- |
| `car.pos`, `car.quat` | `(N, 2P, 3)`, `(N, 2P, 4)` | quat is (w, x, y, z) |
| `car.vel`, `car.angvel` | `(N, 2P, 3)` | angular velocity rotated to world frame |
| `car.steer` | `(N, 2P)` | measured centre-wheel steer angle |
| `car.upright_time` | `(N, 2P)` | seconds spent flipped (for recovery) |
| `ball.pos`, `ball.vel`, `ball.angvel` | `(N, 3)` |  |
| `score` | `(N, 2)` | blue, orange |
| `clock` | `(N,)` | env steps since kickoff of this match |
| `params` | per-world | variant id, latency, trims, friction... (for logging and privileged critics) |
| `measured` | same fields | the noisy copy the agents see (below) |

### `StepEvents` (booleans and counts for this step)

`goal` `(N, 2)`, `ball_touch` `(N, 2P)`, `ball_out` `(N,)`, `kickoff` `(N,)`, `flipped` `(N, 2P)`, `done` `(N,)` (match over), `truncated` `(N,)`. Reward terms and DoneConditions read these instead of re-deriving them; the engine fills the physical events, and the DoneConditions fill done and truncated.

### One step, in order

1. **Latency:** push actions into a ring buffer `(N, Lmax+1, 2P, 2)` and read each world's delayed slot. Sample the delay per match and add per-step jitter, sized from real radio round-trip measurements (Wi-Fi or Bluetooth links commonly add 2–3 ticks).
2. **Motor and ESC model** (`physics.py`), applied to throttle before MuJoCo:
   - deadband: commands with |u| below a threshold produce nothing;
   - response curve: a lookup table from command to effective duty, measured on the real ESC;
   - first-order lag with time constant τ\_motor;
   - voltage sag: stall torque scaled down by a battery-state factor that drops with sustained high current;
   - reverse logic: many RC ESCs brake first and only reverse after a pause at neutral; model it if yours does.
3. **Steering servo:** rate limit, deadband, trim, then the Ackermann split using per-world wheelbase and track.
4. **Ball aerodynamics:** drag `−½ρ Cd A |v| v` plus Magnus `½ρ A r C_L (ω × v)` into `xfrc` at the ball body.
5. **Physics:** `backend.step(ctrl, xfrc, nsub)`.
6. **Read state:** one gather from `qpos` / `qvel` through the `Layout` index tensors.
7. **Rules** (`rules.py`, pure functions): goals, ball touches (from contact sensors), ball out of play, flipped timers, match clock. Results go into `StepEvents`.
8. **Sensors** (`sensors.py`): build `state.measured` from the true state with the failure modes of your tracking system, not just Gaussian noise:
   - per-entity dropouts that come in bursts (a two-state Markov chain), so occlusions last several frames;
   - heavy-tailed position spikes (motion blur, misdetections), for example a small probability of a large jump;
   - timestamp lag and jitter: measurements are delayed by a sampled amount and occasionally arrive out of order;
   - the ball tracked worse than the cars (smaller, faster, often occluded by them).

   Both teams share one measurement, like one overhead camera. Fit every parameter from real logs (Section 15).

### Resets and randomisation are StateMutators

Kickoffs, scattered starts and physics randomisation are StateMutators (Section 4b), listed in the experiment and applied in order to the worlds that need a new start. They write a target `GameState` and per-world parameters; the engine applies them.

```python
class RandomFloorFriction(StateMutator):
    def __init__(self, lo=0.7, hi=1.2): self.lo, self.hi = lo, hi
    def apply(self, state, mask, ctx):
        k = ctx.rand_uniform(mask, self.lo, self.hi)          # (N,) values, used only where mask is True
        ctx.params["floor_friction"] = torch.where(mask, ctx.base.floor_friction * k, ctx.params["floor_friction"])

mutators = [SampleCarVariant(), RandomPhysics.default(), RandomLatency(1, 3),
            Kickoff(formation="standard", scatter_prob=sched.stage(idle=1.0, scripted=0.5, league=0.25))]
```

Mutators run on match start (new match) or on kickoff (after a goal or a dead ball); a mutator that should only run at match start declares `on="match"`. v1's randomisation maps one to one: floor friction, ball mass and bounce, per-car mass and motor strength, steering rate and trim, latency, car variant. On the real cars, only the position-setting mutators apply.

### Truncation and the final observation

When a match ends, `step()` resets that world before returning, so its terminal state would be lost. Keep it: before the reset, copy the `done` worlds' `GameState` into `events.final_state` (the whole batch shape, valid only where `done` is True). The Agents layer turns it into final observations, and the trainer uses them to bootstrap the value of a time-limit truncation. That matters here because matches end on the clock, not on a failure.

## 8. Layer 4: Agents

This layer decides who controls which cars and what each controller sees. `SoccerEnv` (in `env.py`) combines the engine with the components and a roster: it takes one action tensor per agent group, holds it for `action_repeat` ticks, and returns per-group observations and rewards. v1's "paradigms" become two independent choices: the agent granularity (env side) and the network layout (model side), bundled by name in `paradigms.py` (Section 4b).

### Agent granularity (control scheme)

| Scheme | One agent is | Batch per world | Action | Replaces v1 |
| --- | --- | --- | --- | --- |
| `team` | a whole team | 2 | `(6,)` | `team` |
| `car` | one car, seen from its own seat | 2P = 6 | `(2,)` | `shared_car`, and `per_car` (see below) |

v1's `per_car` (three separate networks) is not an env concept any more. With the `car` scheme, each observation carries a `slot` index (0, 1, 2), and the model decides whether to share weights across slots or keep one head per slot (Section 10).

### Roster: who controls each team

Shapes must stay fixed (rule 3), so the roster splits the `N` worlds into fixed slices, set at startup:

| Slice | Worlds | Blue | Orange |
| --- | --- | --- | --- |
| self-play | 0–383 | learner | learner |
| vs scripted | 384–447 | learner | scripted |
| vs league | 448–511 | learner | frozen snapshot k |

The learner's batch is every learner-controlled team in every slice, a fixed count. The league changes which frozen weights a slice uses, never its size. Swapping the slices' proportions over training is done by the curriculum between rollouts, which rebuilds the roster (a rare, outside-the-step operation).

### Observations: entities, not a flat vector

Each agent's observation is a TensorDict of entities, in the team frame, and for the `car` scheme in that car's own frame:

```python
obs = TensorDict({
    "self":      (B, F_self),      # car scheme: own speed, yaw rate, steer, upright
    "slot":      (B,),             # car scheme: which car of the team (0, 1, 2), int64
    "mates":     (B, M, F_car),    # teammates relative to me (M = P-1, or P for team scheme)
    "opponents": (B, P, F_car),
    "ball":      (B, F_ball),      # distance, bearing, relative velocity, height
    "goals":     (B, 2, F_goal),   # own and opponent goal, relative
    "game":      (B, F_game),      # score difference, ...
    "mask":      {"mates": (B, M), "opponents": (B, P)},  # all True in 3v3
    "critic":    (B, F_priv),      # privileged, noise-free state for the value net
}, batch_size=[B])
```

Why entities: a flat-vector MLP can still concatenate them, but a transformer or set encoder can treat each car as a token, and the same model then works for 2v2 or 4v4 because masks absorb the count. v1's ego features (Section 2, `features.py`) become the per-entity feature functions.

**Privileged critic:** the value network can see the true, noise-free state, the per-world parameters and the opponent's identity, because it is only used in training. This usually helps a lot when observations are noisy, and costs nothing at deployment.

### Observation terms are registered and configured

```python
ObsSpec(
    scheme="car",
    self_terms=["fwd_speed", "lat_speed", "yaw_rate", "steer", "upright", "slot"],
    car_terms=["rel_pos", "rel_vel", "rel_heading", "distance"],
    ball_terms=["rel_pos", "rel_vel", "height", "distance"],
    source="estimated",  # actor reads the estimator's output (Section 15); critic reads the true state
    history=0,           # >0 stacks past frames (for MLPs without memory)
)
```

Each term is a pure function `(GameState, frames) → Tensor`. The same functions run on the real robots: build a `GameState` from your tracking system and call them, so training and deployment can't drift apart.

### Frames (`frames.py`)

- **Team frame:** orange's coordinates are rotated 180° about z (x, y negated, yaw + π), so every agent attacks toward +x and one policy plays both sides.
- **Ego frame:** a vector `(dx, dy)` rotated by the car's yaw: `(c·dx + s·dy, −s·dx + c·dy)`, with `c, s = cos yaw, sin yaw`.

### Action repeat lives here

`SoccerEnv.step(actions)` parses every group's actions into `car_actions (N, 2P, 2)`, calls `engine.step` `action_repeat` times, sums rewards and ORs the events across those ticks, and drops reward that arrives after a match ended mid-repeat. Policies and the trainer only ever see decisions.

### The v1 contract as an adapter

`flat87(state, team)` rebuilds v1's 87-number vector from `GameState`. Keep it for comparison with v1 checkpoints and for any controller code already written against it.

## 9. Layer 5: Rewards

A reward is written as a weighted sum of named terms, like the maths, and any weight can be a schedule instead of a number. Terms are small pure functions; the framework handles scopes (per car vs per team), scheduling, logging and conversion to whatever agent granularity the run uses.

### What writing a reward looks like

```python
from car_soccer.rewards import term, sched, potential
from car_soccer.rewards.terms import goal, conceded, ball_to_goal_vel, crowding, flipped

@term(scope="car", units="m/s")
def approach_ball(ctx):
    to_ball = unit(ctx.ball.pos[:, None, :2] - ctx.car.pos[..., :2])   # (N, 2P, 2)
    return (ctx.car.vel[..., :2] * to_ball).sum(-1)                     # (N, 2P)

reward = (
      10 * goal
    - 10 * conceded
    + sched.stage(idle=1.0, scripted=0.5, league=0.2) * approach_ball.nearest()
    + sched.linear(0.3, 1.0, start="0", end="5M") * ball_to_goal_vel
    - 0.2 * crowding
    - 0.5 * flipped
)
```

That expression is the whole reward definition. It goes in the experiment file (Section 11), so changing a reward never means editing the env.

### The pieces

**`ctx` (what a term can read):** `ctx.car`, `ctx.ball`, `ctx.score` (true state), `ctx.prev` (last decision's state), `ctx.events` (goals, touches, kickoffs, flips), `ctx.actions` and `ctx.prev_actions`, `ctx.dt`, `ctx.team_sign`. Everything is a tensor in the world frame; helpers give team-frame views.

**Scope:** a term returns `(N, 2P)` if `scope="car"` or `(N, 2)` if `scope="team"`. Terms are written once, from "my" point of view; the framework evaluates them for both teams.

**Weights:** a number, or a schedule (a function of training progress):

| Schedule | Example | Value |
| --- | --- | --- |
| `sched.const(v)` | `sched.const(0.5)` | always v |
| `sched.linear(a, b, start, end)` | `sched.linear(1.0, 0.1, "0", "10M")` | a to b over env steps, then held |
| `sched.piecewise({...})` | `sched.piecewise({"0": 1.0, "2M": 0.5, "8M": 0.0})` | steps between keys |
| `sched.stage(...)` | `sched.stage(idle=1.0, league=0.2)` | by curriculum stage name |
| `sched.cosine(a, b, start, end)` |  | smooth decay |
| `sched.when(metric, op, x, then, else_)` | `sched.when("win_rate_scripted", ">", 0.6, 0.0, 0.5)` | switches on a logged metric |
| any `callable(progress) -> float` |  | your own |

`progress` is one object: `step`, `stage`, and the latest logged `metrics`. Schedules are evaluated once per rollout on CPU, so a weight is a Python float inside the step and costs nothing on the GPU.

**Modifiers** return a new term:

| Modifier | Effect |
| --- | --- |
| `.nearest()` | keep only each team's car nearest the ball (zero for the others) |
| `.team_sum()`, `.team_mean()` | car scope to team scope |
| `.opponent()` | the other team's value of this term |
| `.zero_sum()` | mine minus the opponent's |
| `.clip(lo, hi)`, `.scale(k)` | bound or rescale |
| `.rate()` | multiply by `dt` so a per-second quantity sums correctly over a decision |
| `potential(phi)` | potential-based shaping: `γ·φ(s') − φ(s)`, which provably leaves the optimal policy unchanged |

Prefer `potential(...)` for shaping such as "get closer to the ball" or "ball closer to the goal": plain velocity rewards can be farmed (circle the ball, push it back and forth), potential-based ones can't.

**Keep potentials bounded.** Normalise distances by the field diagonal so φ stays in \[−1, 0\]. With γ < 1 the shaping term equals (φ' − φ) − (1 − γ)·φ', and the second part accumulates: over a 60 s match (about 1,000 decisions at γ = 0.99) it can reach 10 × its weight, as large as a goal. The optimal policy is unchanged in theory, but learning feels it, so the report logs each term's per-match total. Keep every shaping total well below the goal reward.

### From terms to the reward each agent gets

The `RewardManager` evaluates the expression once per decision for all worlds and both teams, then maps it to the agent granularity:

- **Team agents:** car-scope terms are summed over the team (or averaged, by config); team-scope terms pass through.
- **Car agents:** team-scope terms are shared by all three cars; car-scope terms are blended with the team average using a `team_spirit` weight τ, `r_i = (1 − τ) · r_i + τ · mean(r_team)`. τ can itself be scheduled, typically 0.3 early and 1.0 late, so cars learn individually first and play as a team later.
- **Logging:** for every named term, the raw mean and the weighted contribution per decision go to the logs automatically, so you can see which term is driving behaviour.

### Built-in terms (`rewards/terms.py`)

| Term | Scope | Meaning |
| --- | --- | --- |
| `goal`, `conceded` | team | 1 on the decision a goal is scored / let in |
| `ball_to_goal_vel` | team | ball velocity toward the opponent goal, m/s |
| `approach_ball` | car | own velocity toward the ball, m/s |
| `ball_touch` | car | 1 when this car touched the ball |
| `ball_dist_potential` | car | − distance to the ball, for `potential()` |
| `ball_goal_potential` | team | − ball distance to the opponent goal, for `potential()` |
| `crowding` | team | teammate pairs closer than 1 m |
| `flipped` | car | 1 while on side or roof |
| `action_rate` | car | change in action since the last decision (smoothness) |
| `throttle_energy` | car | throttle squared (battery, motor heat) |

## 10. Layer 6: Models

Every policy is a pipeline of four swappable parts: encoder, core, action head, value head. Each part is a registered `nn.Module` with a fixed input and output contract, so an MLP, a transformer over entities, or a recurrent memory plugs in by changing one line of the experiment file. Yes, you hand-write each part, but each is 20–80 lines and the trainer never changes.

### Contract

```python
@dataclass
class ModelOut:
    dist: torch.distributions.Distribution   # sample(), log_prob(a) summed over action dims, entropy()
    value: Tensor                            # (B,)
    hidden: TensorDict | None                # recurrent state, None for feed-forward models

class AgentModel(nn.Module):
    def forward(self, obs: TensorDict, hidden: TensorDict | None = None,
                done: Tensor | None = None) -> ModelOut: ...
    def initial_hidden(self, batch_size: int) -> TensorDict | None: ...
    def export_actor(self, path: str) -> None: ...   # deployment: actor + normaliser only
```

### The four parts

| Part | Input → output | Built-ins to write |
| --- | --- | --- |
| Encoder | obs TensorDict → `(B, D)` | `flat_mlp`: concatenate all entities, MLP. `deepset`: one MLP per entity type, mean and max pooled. `entity_transformer`: each car, the ball and the goals become tokens; 2–3 self-attention layers; output the "self" token. |
| Core (memory) | `(B, D)`, hidden → `(B, D')`, hidden | `identity`, `gru`, `lstm`. Resets hidden where `done`. |
| Action head | `(B, D')` → distribution | `gaussian` (learned log-std, clipped to \[−1, 1\] in the env), `squashed_gaussian` (tanh), `beta` (naturally bounded), `bins` (e.g. 5 throttle × 7 steer categorical). |
| Value head | critic obs → `(B,)` | Its own encoder, usually `flat_mlp` on the privileged `critic` entry. |

Per-slot networks (v1's `per_car`): `slots="shared"` uses one network for all cars; `slots="heads"` shares the encoder and gives each slot its own action head (selected by `obs["slot"]`); `slots="separate"` builds three full pipelines. All three take the same batch.

### Configuring a model

```python
ModelSpec(
    encoder=("entity_transformer", dict(d_model=96, layers=2, heads=4)),
    core=("gru", dict(hidden=128)),
    action=("gaussian", dict(init_std=0.6, throttle_bias=0.3)),
    critic=dict(encoder=("flat_mlp", dict(hidden=(256, 256))), obs_key="critic"),
    slots="shared",
    normalize_obs=True,
)
```

Adding your own architecture:

```python
@register_encoder("my_gnn")
class MyGNN(nn.Module):
    def __init__(self, obs_spec: ObsSpec, d_out: int, **kw): ...
    def forward(self, obs: TensorDict) -> Tensor:   # (B, d_out)
        ...
```

Then `encoder=("my_gnn", {...})` in the experiment file. The registry builds the module from the observation spec, so input sizes are never hard-coded.

### Details that mattered in v1

- **Normaliser inside the model:** running mean and std per feature, updated during collection only, saved in the checkpoint and exported with the actor.
- **Initialisation:** orthogonal weights, action-head output gain 0.01, value-head gain 1.0.
- **Throttle prior:** start the throttle mean at +0.3 so early exploration actually moves the cars.
- **Log-std** clamped to \[−3, 0.5\].

### Speed notes

- Collection runs under `torch.inference_mode()` with one forward pass for every learner agent in the batch.
- A transformer over about 8 tokens at a batch of a few thousand agents is cheap; on CPU keep `d_model` at 64–128.
- Use `torch.compile` on the model once it is stable.
- Recurrent cores need sequences in training (Section 11) and carry hidden state on the robot at deployment too.

## 11. Layer 7: Training

A training run is one experiment file, a Python object that names every choice from the layers below. The runner wires those choices together; algorithms, league and curriculum are separate parts with small interfaces.

### Library choice: hand-written PPO, not RSL-RL

This changes my earlier recommendation. RSL-RL is fast and minimal, but your new requirements (entity TensorDict observations, recurrent and transformer models, frozen opponents inside the batch, a privileged critic, scheduled coefficients) would each need working around its assumptions. A \~300-line PPO behind the `Algorithm` interface below is easier to bend, and the RL library is never the bottleneck here: the simulator is. Keep RSL-RL (or CleanRL's PPO) as a reference: run the same simple task on both once and check that the learning curves roughly match. If you'd rather not write the loss yourself, TorchRL's `ClipPPOLoss` and GAE modules work with TensorDicts and slot into the same interface.

### Algorithm interface

```python
class Algorithm(Protocol):
    model: AgentModel
    def act(self, obs: TensorDict, hidden) -> tuple[Tensor, TensorDict]:
        """Sample actions; extras hold log-prob, value, new hidden (stored in the buffer)."""
    def update(self, batch: TensorDict, progress: Progress) -> dict[str, float]:
        """One learning step on a full rollout; returns metrics to log."""
    def state_dict(self) -> dict: ...
    def load_state_dict(self, d: dict) -> None: ...
```

### PPO, as it should work here

- GAE with γ = 0.99 per decision (about 6 s at 17 Hz) and λ = 0.95.
- **Bootstrapping at time limits:** for a world truncated by the match clock, add `γ · V(final_obs)` to its last reward instead of treating it as a true end.
- Clipped objective (0.2), clipped value loss, advantage normalisation per minibatch, KL early stop (0.03), gradient-norm clip 0.5.
- **Reward normalisation:** divide rewards by a running std of the discounted return (Section 3).
- Learning rate, entropy coefficient and clip range accept schedules from `sched` (Section 9).
- **Recurrent models:** split each rollout into sequences of 16–32 decisions, store the hidden state at each sequence start, and keep sequences whole when shuffling minibatches.

### Collector and buffer

The collector runs `T` decisions (for example 64) across all worlds and stores only learner agents' data in a `TensorDict` shaped `(T, B_learner)`: `obs`, `action`, `log_prob`, `value`, `reward`, `done`, `truncated`, `final_value`, and `hidden` for recurrent cores. Opponents act inside `SoccerEnv` through the roster and never enter the buffer.

### League

- **Pool:** a frozen copy of the learner every `snapshot_every` iterations, up to `pool_size`, plus fixed members (scripted bots, idle cars, named checkpoints).
- **Matchmaking:** each rollout, the league picks which snapshot each league slice faces. Uniform sampling works; better is prioritised fictitious self-play, which favours snapshots the learner still loses to (weight ∝ its loss rate against each).
- **Ratings:** short evaluation matches between members update an Elo or TrueSkill table, which gives one number for "is the agent improving" even when all play is self-play.

### Curriculum

A curriculum is a list of stages; each names its own exit condition, roster mix and env overrides. Stage names are what `sched.stage(...)` reads.

```python
Stage("idle",     until=steps("5M") | metric("approach_speed", ">", 0.5),
      roster={"vs_idle": 1.0},                         env={"random_kickoff": 1.0, "match_seconds": 10})
Stage("scripted", until=metric("win_rate_scripted", ">", 0.6),
      roster={"vs_scripted": 0.8, "vs_idle": 0.2},     env={"random_kickoff": 0.5, "match_seconds": 30})
Stage("league",   until=steps("200M"),
      roster={"self": 0.6, "league": 0.25, "vs_scripted": 0.15}, env={"random_kickoff": 0.25, "match_seconds": 60})
```

### The experiment file (one per run)

```python
# experiments/league_transformer.py
from car_soccer.api import *

experiment = Experiment(
    name="league_transformer",
    scene=SceneConfig(ball=BallConfig(radius=8 * INCH), variants=VariantConfig(pool_size=16)),
    game=GameConfig(control_hz=50, action_repeat=3, noise=NoiseConfig.realistic()),
    agents=AgentsConfig(scheme="car", obs=ObsSpec.default_car()),
    reward=10 * goal - 10 * conceded
           + sched.stage(idle=1.0, scripted=0.5, league=0.1) * potential(ball_dist_potential).nearest()
           + sched.linear(0.3, 1.0, "0", "20M") * potential(ball_goal_potential)
           - 0.2 * crowding,
    team_spirit=sched.linear(0.3, 1.0, "0", "50M"),
    model=ModelSpec(encoder=("entity_transformer", dict(d_model=96, layers=2)), core=("gru", dict(hidden=128))),
    algo=PPOConfig(lr=sched.linear(3e-4, 5e-5, "0", "200M"), rollout=64, epochs=4, minibatches=8),
    league=LeagueConfig(snapshot_every=50, pool_size=20, matchmaking="pfsp"),
    curriculum=[...],                          # the stages above
    run=RunConfig(num_worlds=1024, device="cpu", backend="cpu_rollout", seed=0, total_steps="200M"),
)
```

Run it with `python scripts/train.py experiments/league_transformer.py --set run.num_worlds=512`. The checkpoint stores the model, optimiser, normalisers, league pool, ratings, curriculum stage, progress, and a copy of the experiment file's source, so `--resume` continues exactly where it stopped.

## 12. Evaluation, logging and tooling

Evaluation always runs the same way: fixed seeds, deterministic actions, a named opponent, and a fixed number of full matches. That way two checkpoints can be compared fairly.

### Evaluation

- `evaluate(model, opponent, matches=64, conditions="realistic" | "nominal", car="variants" | "nominal" | path)` returns win, draw and loss rates; goals for and against; ball time in the opponent half; mean ball x; car speed; touches per car.
- **Opponent ladder:** idle, scripted, then frozen checkpoints by name, so progress reads as "beats scripted 70%, beats last week's model 55%".
- **Robustness sweep:** the same evaluation per car variant and per noise level. A car-invariant policy should hold its win rate across variants.
- **Video:** `record(model, opponent, out="match.mp4")` renders one world offscreen (`MUJOCO_GL=egl` on headless Linux, the default renderer on macOS).

### Logging

- Every iteration: throughput (env steps per second, split into physics, policy and update time), each reward term's raw and weighted mean, PPO diagnostics (KL, clip fraction, entropy, explained variance), curriculum stage, league ratings.
- Every `eval_every` iterations: the evaluation above against the ladder.
- Back-ends: CSV always; TensorBoard or Weights & Biases optional through one `Logger` interface.

### Scripts

| Script | Does |
| --- | --- |
| `train.py <experiment> [--set k=v] [--resume ckpt]` | train |
| `evaluate.py <ckpt> --opponent <name or ckpt>` | evaluation report |
| `bench.py --worlds 512 --backend cpu_rollout` | env steps per second, with and without the policy |
| `inspect_car.py <urdf or json>` | car roles, measured geometry, 4-view render (from v1) |
| `record.py <ckpt>` | match video |
| `plot.py runs/a runs/b` | compare runs |

### Tests (`tests/`, run with `pytest`)

- Scene: model compiles; variants share structure; probe catches a rotated car.
- Backend: v1's 16 physics checks; CPU vs GPU parity (once GPU exists).
- Game: a goal fires exactly once and kicks off; flip recovery; latency delays actions by the right number of ticks; no per-world Python loop (time `step()` at `N` = 64 vs 1024: the per-world cost must fall).
- Agents: team-frame symmetry (both teams see a mirrored kickoff identically); `flat87` matches v1.
- Rewards: each built-in term's sign on a hand-built state; schedules at their breakpoints; a potential term sums to φ(end) − φ(start) over an episode (with γ = 1).
- Models: every registered encoder accepts the observation spec; recurrent hidden resets on `done`.
- PPO: learns the trivial "maximise throttle" reward in under 20 iterations (Section 3).

## 13. Build plan

Build bottom-up, one layer per milestone, and don't start the next until its done-when test passes. v1 stays in the repo as the reference: several tests compare against it.

| # | Milestone | You learn | Done when |
| --- | --- | --- | --- |
| M0 | Setup: repo, venv, `pytest`, `mujoco`, `torch`, `tensordict`; run v1 | Project tooling | v1's `physics_check.py` passes 16/16 on your machine and you've recorded v1's env steps per second |
| M1 | Scene: port the model builder; `Layout` as tensors; collision groups; variants as parameter packs | MJCF, `MjSpec`, contacts, actuators | `test_scene.py` passes; the scene opens in the MuJoCo viewer |
| M2 | CPU backend on `mujoco.rollout` | Threads, the GIL, state vectors | v1's physics checks pass through the new backend; physics-only speed rises with `nthread` |
| M3 | Game: `GameState`, actuation, aerodynamics, rules, sensors, flip recovery (the engine); StateMutators for kickoffs and randomisation; DoneConditions; scripted bots ported to torch | Vectorised logic with masks; car dynamics | `test_game.py` passes; scripted bots finish a 60 s match; full game step costs at most \~15% more than physics alone |
| M4 | Agents: frames, entity observation terms, schemes, roster, `SoccerEnv with action repeat`, `flat87` | Coordinate frames, multi-agent batching | Symmetry test passes; `flat87` equals v1's observation on the same state |
| M5 | Rewards: `@term`, expression tree, modifiers, schedules, `potential`, `RewardManager` | Reward shaping theory | `test_rewards.py` passes; v1's staged reward reproduced exactly |
| M6 | Models: registry, `flat_mlp`, Gaussian head, value head, normaliser, `ModelSpec` | PyTorch module design | `test_models.py` passes; `export_actor` round-trips |
| M7 | PPO and runner: collector, buffer, GAE with truncation bootstrap, update, reward normalisation, logging, checkpoints | PPO, end to end | Trivial throttle task learns in under 20 iterations; then v1's quick result is matched (at 2M steps: ≥ 75% draws vs scripted, > 10 goals per match better than untrained) |
| M8 | Curriculum and league: stages, roster rebuild, snapshot pool, prioritised matchmaking, Elo | Self-play dynamics | A run passes through every stage; the latest snapshot's Elo beats earlier ones |
| M9 | Advanced models: `deepset`, `entity_transformer`, `gru` with sequence minibatches, slot heads | Attention, recurrence in RL | Each trains on the approach task; learning curves compared in one plot |
| M10 | Speed pass: 2 ms timestep check, contact settings, `torch.compile`, profiling | Profiling, performance trade-offs | `bench.py` meets your target on the cloud instance |
| M11 | (later) GPU backend | GPU simulation | Section 14 checklist |

The real-world track runs alongside M8–M11 once the cars exist:

- **M12, estimator and sensor realism:** the Kalman filter on the shared path, and simulated dropouts in bursts, position spikes and timestamp jitter. Done when the estimator's error in sim matches what you measure on the real tracking system.
- **M13, motor model and system identification:** deadband, response curve, lag, voltage sag; `sysid.py` fitting real logs. Done when sim and real overlay plots agree for every Section 15 test drive.
- **M14, RealEngine and deployment:** tracking, radio, safety, positioning, policy runner. Done when each rung of the deployment ladder passes.

A good rhythm with the tutor: read the layer's section, write the contract (signatures and shapes) first, ask the tutor to review it against the guide, then fill in the code and run the test.

## 14. CPU to GPU migration

If every layer above the backend follows the rules in Section 4, moving to GPU means writing one new backend and changing `device` and `backend` in the run config. Work through this list in order.

- [ ] Export the scene XML and time it with MuJoCo Warp's speed test on the target GPU before writing any code; contact-heavy scenes like six cars, a ball and walls are where GPU simulators slow down.
- [ ] Check which model fields your MuJoCo Warp version can batch per world (friction, masses, gains, geometry). Car variants need geometry; if it can't be batched, assign each variant a fixed block of worlds.
- [ ] Write `sim/warp.py` implementing `SimBackend`; capture the `nsub` substeps as one CUDA graph.
- [ ] Pass the parity test (Section 6) and v1's physics checks on the GPU backend.
- [ ] Grep the step path for `.item()`, `.cpu()`, `.numpy()`, Python `if` on tensors and per-world loops; each one is a GPU sync or a slowdown.
- [ ] Move the random generator, roster masks and latency ring buffer to the GPU (they are created on `env.device`, so this should need no code changes).
- [ ] Raise `num_worlds` (thousands) and the PPO minibatch size to match.
- [ ] Profile one iteration: physics, observations and rewards, policy forward, update. Fix the largest first.
- [ ] Compare learning curves on CPU and GPU for the same experiment and seed range; they should overlap within noise.

## 15. Sim to real

The real cars are one more engine: `RealEngine` implements the Section 7 contract with `num_worlds = 1`, and every component above it (estimator, ObsBuilder, ActionParser, the exported policy) is the same code that ran in training. What transfers or doesn't is decided by how well the simulator's imperfections match the real ones, so system identification comes before any real match.

### RealEngine parts (`engine/real/`)

| Part | Does |
| --- | --- |
| `tracking.py` | Reads mocap or camera detections with timestamps and turns them into raw measurements, the same format `sensors.py` produces in sim |
| `radio.py` | Sends `[throttle, steer]` to each car every tick with a sequence number; measures round-trip time |
| `safety.py` | Watchdog (zero throttle if no command for 100 ms), geofence, `throttle_limit`, keyboard e-stop |
| `positioning.py` | `set_state`: a scripted controller drives each car to its kickoff target; an operator places the ball; returns when everything is within tolerance |
| `real_engine.py` | `step()`: send commands, wait for the next 20 ms tick, read tracking, detect goals and touches |

### Put the state estimator on the shared path

The policy should never read raw measurements, in sim or real. A shared `components/estimator.py` (a constant-velocity Kalman filter per car and for the ball) fills dropouts, rejects spikes and estimates velocities. It runs on `sensors.py` output in training and on `tracking.py` output on the cars, so the policy learns the estimator's real lag and errors. This is the single most important step for transfer: velocities computed from a noisy camera are far worse than simulated ones.

### System identification before training for real

1. **Log real drives** with the commands sent: full-throttle straight line, coast-down from speed, step steer at several speeds, constant-steer circles, ball drops on the real floor and off the real walls, a car pushing the ball.
2. **Replay** the logged commands in `SimEngine` with hundreds of parameter sets at once (one per world: friction, motor lag and deadband, stall torque, steer rate, restitution, latency).
3. **Fit** the parameters that best reproduce the logged trajectories (`scripts/sysid.py`; CMA-ES or a simple random search).
4. **Set** the nominal config to the fit, and the randomisation ranges to the fit plus its uncertainty.
5. **Check** with overlay plots of sim vs real for each test, kept in `tests/` as a regression.

### Deployment ladder

1. One car drives to points on the field (positioning controller, no policy).
2. One car, policy, no opponents: push the ball into the goal, `throttle_limit` 0.3.
3. 1v1 against a scripted car, then a policy car.
4. 3v3, raising `throttle_limit` step by step.

At each rung, record the same metrics as the sim evaluation (touches, goals, ball time in the opponent half) and compare against sim under realistic noise; a big gap points back to system identification.

### Running the policy on the cars

`deploy/export.py` saves the actor and its normaliser as TorchScript or ONNX. `deploy/policy_runner.py` runs the estimator, the ObsBuilder and the actor once per decision (about 17 Hz) on a laptop or a Jetson, and holds each command for `action_repeat` ticks, exactly as in training.

## 16. Open decisions

These are the choices I made on your behalf; each is worth confirming before M1.

| Decision | Current default | Alternatives |
| --- | --- | --- |
| Ball size | 8 in radius (16 in diameter) | 4 in radius (8 in diameter): smaller goals and walls, 1 ms timestep still fine |
| Tensor library above the backend | PyTorch + TensorDict | numpy on CPU (faster small ops, but a rewrite for GPU); JAX (needs MJX, poor on your hardware) |
| RL algorithm code | Hand-written PPO behind `Algorithm` | RSL-RL; TorchRL losses; PufferLib |
| Config format | Python experiment files | YAML plus registries (easier to diff, can't hold reward expressions directly) |
| Agent granularity default | `car` scheme, shared weights | `team` scheme |
| Critic | Privileged (true state + world parameters) | Same observation as the actor |
| Decision rate | 50 Hz control, decision every 3 ticks | Match your real loop once known |
| Real-world sensing | One overhead camera or motion capture, shared by both teams | On-board sensors per car (would change observations a lot) |
| Cloud machine | Google Cloud instance you have access to | Spot `c3d-highcpu-60` for CPU; `g2-standard-8` (L4) after the GPU backend exists |

### References

- [rsl-rl-lib on PyPI](https://pypi.org/project/rsl-rl-lib/5.2.0/), the reference PPO for M7.
- [mujoco-warp on PyPI](https://pypi.org/project/mujoco-warp/3.5.0/), the GPU backend for M11.
- [c3d-highcpu-60 pricing](https://gcloud-compute.com/c3d-highcpu-60.html) and [g2-standard-8 pricing](https://gcloud-compute.com/g2-standard-8.html).
- MuJoCo's `rollout` module docstring (`help(mujoco.rollout.rollout)` in Python) for the CPU backend's arguments.
- v1 docs in the repo: `docs/TRAINING.md`, `docs/CAD_TO_URDF.md`.
