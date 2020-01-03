from ModelRunner import ModelRunner
from model import BoidFlockers


param_path = "./parameter/nominal.toml"
runner = ModelRunner(BoidFlockers, param_path)
runner.run_headless()
