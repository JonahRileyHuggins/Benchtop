import os
import sys
sys.path.append('bin/')
sys.path.append('Benchtop/')
from src.benchtop.Experiment import Experiment
from wrappers.sparced_wrapper import WrapSPARCED
config_path = "Benchtop/tests/BIM-dependent-ERK-inhibition/config.yml"
experiment = Experiment(config_path, cores=os.cpu_count() , verbose=True)
# 2025-10-07 09:21:32,928 - INFO - Loading Experiment None details from /SPARCED/Benchtop/tests/BIM-dependent-ERK-inhibition/config.yml
experiment.run(WrapSPARCED, ("./SPARCED.xml", "./SPARCED/", 0))