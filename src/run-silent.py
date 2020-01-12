from model import BoidFlockers
from ModelRunner import ModelRunner

param_path = "./parameter/nominal.toml"
runner = ModelRunner(BoidFlockers, param_path)
runner.run_silent()
