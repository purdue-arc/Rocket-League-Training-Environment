def testImports():
  import carSoccer.config as cfg, mujoco, torch, tensordict

def testBallConfigDefaults():
  import carSoccer.config as cfg
  config = cfg.BallConfig()

  assert config.radius > 0

def testMujocoSteps():
  import mujoco
  
  model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
  data = mujoco.MjData(model)
  
  assert data.time == 0.0
  
  mujoco.mj_step(model, data)
  
  assert data.time == model.opt.timestep