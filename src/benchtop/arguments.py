"""Argument parser for the benchtop CLI."""

import argparse
import os

ASCII_HEADER = r"""
 ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄         ▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░▌      ▐░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░▌░▌     ▐░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░▌       ▐░▌ ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌
▐░▌       ▐░▌▐░▌          ▐░▌▐░▌    ▐░▌▐░▌          ▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌▐░▌       ▐░▌
▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌ ▐░▌   ▐░▌▐░▌          ▐░█▄▄▄▄▄▄▄█░▌     ▐░▌     ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌
▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░▌  ▐░▌  ▐░▌▐░▌          ▐░░░░░░░░░░░▌     ▐░▌     ▐░▌       ▐░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░▌   ▐░▌ ▐░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌     ▐░▌     ▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀▀▀ 
▐░▌       ▐░▌▐░▌          ▐░▌    ▐░▌▐░▌▐░▌          ▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌▐░▌          
▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌     ▐░▐░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌     ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          
▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░▌      ▐░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌     ▐░▌     ▐░░░░░░░░░░░▌▐░▌          
 ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀        ▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀       ▀       ▀▀▀▀▀▀▀▀▀▀▀  ▀           
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=ASCII_HEADER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    shared_parser = argparse.ArgumentParser(add_help=False)
    global_group = shared_parser.add_argument_group("Global Options")
    global_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="verbose",
        help="Enable verbose logging.",
    )
    global_group.add_argument("-p", "--path", help="Path to benchmark YAML.")
    global_group.add_argument("-n", "--name", help="Descriptive name for this run.")
    global_group.add_argument(
        "-o", "--output",
        default=".",
        help="Output directory (default: current directory).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    experiment_parser = subparsers.add_parser(
        "experiment",
        parents=[shared_parser],
        help="Run PEtab-style simulation experiments",
    )
    exp_group = experiment_parser.add_argument_group("Experiment Options")

    exp_group.add_argument(
        "--simulator", "-s",
        type=str,
        default="tellurium",
        help="Simulation backend (tellurium, amici, bngsim).",
    )
    exp_group.add_argument(
        "--cores", "-c",
        type=int,
        default=os.cpu_count(),
        help=f"Parallel processes (default: {os.cpu_count()}).",
    )
    exp_group.add_argument(
        "--cache_dir",
        type=str,
        default="./.cache",
        help="Cache directory for simulation trajectories.",
    )
    exp_group.add_argument(
        "--load_index",
        type=bool,
        default=False,
        help="Resume from an existing cache index.",
    )
    exp_group.add_argument(
        "--No_Observables",
        action="store_true",
        help="Skip observable calculation after simulation.",
    )
    exp_group.add_argument(
        "--catchall",
        metavar="KEY=VALUE",
        nargs="*",
        help="Additional experiment arguments as key=value pairs.",
    )
    exp_group.add_argument(
        "--run_all",
        default=None,
        help="Run all benchmarks in the given directory.",
    )

    return parser.parse_args()
