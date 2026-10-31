import mujoco
import pytest

import carSoccer.config as cfg
from carSoccer.engine.sim.scene import ball

@pytest.fixture
def makeModel():
  """Factory: compile the ball-drop scene from (optional) configs."""
  def _make(ballCfg=None, simCfg=None, dropHeight=1.0):
    ballCfg = ballCfg or cfg.BallConfig()
    simCfg = simCfg or cfg.SimConfig()
    xml = ball.ballDropXML(ballCfg, simCfg, dropHeight)
    return mujoco.MjModel.from_xml_string(xml)
  
  return _make

@pytest.fixture
def model(makeModel):
  """The default ball-drop model."""
  return makeModel()