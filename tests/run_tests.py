import os
import sys
sys.path.append('tests/')
sys.path.append('src/benchtop/')
sys.path.append(os.path.dirname(__file__))

os.chdir('../')

def run_tests() -> None:
    import test_benchtop
    test_benchtop.test_run()
    test_benchtop.test_reassigning_all_species()

    import test_worker
    test_worker.test_worker_constructor()
    test_worker.test_find_preequilibration_results()
    test_worker.test_find_preequilibration_results_no_match()
    test_worker.test_setModelState_basic()
    test_worker.test_get_simulation_time()
    test_worker.test_model_state_assignment()

if __name__ == '__main__':

    run_tests()