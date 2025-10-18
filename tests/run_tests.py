import os
import sys
sys.path.append('tests/')
sys.path.append('src/benchtop/')
sys.path.append(os.path.dirname(__file__))

import logging

logging.basicConfig(
    level=logging.DEBUG, # Overriden if Verbose Arg. True
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.chdir('../')

def run_tests() -> None:
    import test_benchtop
    test_benchtop.test_results_dict_inheritance()
    test_benchtop.test_results_saving()
    test_benchtop.test_run()
    # test_benchtop.test_reassigning_all_species()

    import test_worker
    test_worker.test_worker_constructor()
    test_worker.test_find_preequilibration_results()
    test_worker.test_find_preequilibration_results_no_match()
    test_worker.test_setModelState_basic()
    test_worker.test_get_simulation_time()
    test_worker.test_model_state_assignment()

    import test_organizer
    test_organizer.test_organizer_constructor()
    test_organizer.test_topological_sorter()
    test_organizer.test_delay_secondary_condition()
    test_organizer.test_total_tasks_basic()
    test_organizer.test_total_tasks_empty_tasks()
    test_organizer.test_total_tasks_zero_cells()
    

if __name__ == '__main__':

    run_tests()