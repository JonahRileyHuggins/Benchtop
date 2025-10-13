import os
import sys
sys.path.append('tests/')
sys.path.append('src/benchtop/')
sys.path.append(os.path.dirname(__file__))

os.chdir('../')

def run_tests() -> None:
    from test_benchtop import test_run
    test_run()


if __name__ == '__main__':

    run_tests()