import os
import sys
sys.path.append('./build')
sys.path.append('./python/sdist/singlecell/Benchtop/')
sys.path.append('./python/sdist/singlecell/Benchtop/src')
from benchtop.Experiment import Experiment
from wrappers.SingleCell import SingleCell
config_path = "benchmarks/BIM-dependent-ERK-inhibition/BIM-dependent-ERK-inhibition.yml"
experiment = Experiment(config_path, cores=os.cpu_count() , verbose=True)
## 2025-10-10 12:45:52,729 - INFO - Loading Experiment None details from /SingleCell/benchmarks/BIM-dependent-ERK-inhibition/BIM-dependent-ERK-inhibition.yml
## 2025-10-07 09:21:32,928 - INFO - Loading Experiment None details from /SPARCED/Benchtop/tests/BIM-dependent-ERK-inhibition/config.yml
experiment.run(SingleCell, ("sbml_files/stochastic.xml", "sbml_files/deterministic.xml"))
experiment.observable_calculation()
# experiment.run(WrapSPARCED, ("./SPARCED.xml", "./SPARCED/", 0))

sys.path.append('bin/')
sys.path.append('Benchtop/')
from src.benchtop.Experiment import Experiment
from wrappers.sparced_wrapper import WrapSPARCED
config_path = "Benchtop/tests/BIM-dependent-ERK-inhibition/config.yml"
experiment = Experiment(config_path, cores=os.cpu_count() , verbose=True)
#2025-10-07 09:21:32,928 - INFO - Loading Experiment None details from /SPARCED/Benchtop/tests/BIM-dependent-ERK-inhibition/config.yml
experiment.run(WrapSPARCED, ("./SPARCED.xml", "./SPARCED/", 0))
experiment.observable_calculation()