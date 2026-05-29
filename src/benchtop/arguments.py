#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================
# ============ Package Import ============
# =========================================
import os
import argparse

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

# =========================================
# ============ CLI Arguments ==============
# =========================================
def parse_args():
    """Retrieve and parse arguments necessary for model creation"""

    parser = argparse.ArgumentParser(
        description=ASCII_HEADER,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ---------- Shared / Global Options ----------
    shared_parser = argparse.ArgumentParser(add_help=False)
    global_group = shared_parser.add_argument_group("Global Options")
    global_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest="verbose",
        help="Enable verbose logging output."
    )
    global_group.add_argument(
        '-p', '--path',
        help="Path to data file."
    )
    global_group.add_argument(
        '-n', '--name',
        help="Descriptive name for this run."
    )
    global_group.add_argument(
        '-o', '--output',
        default=".",
        help="Directory to store output files (default: current directory)."
    )

    # ---------- Subcommands ----------
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =========== [Command: Experiment] ======
    experiment_parser = subparsers.add_parser(
        "Experiment",
        parents=[shared_parser],
        help="Execute simulation experiments in PEtab Problem format"
    )
    exp_group = experiment_parser.add_argument_group("Experiment Options")
    
    exp_group.add_argument(
        '--simulator', '-s',
        type=str, 
        default="tellurium",
        help=f"Simulation engine to use for simulation experiment."
    )
    exp_group.add_argument(
        '--cores', '-c',
        type=int,
        default=os.cpu_count(),
        help=f"Number of parallel processes (default: {os.cpu_count()})."
    )
    exp_group.add_argument(
        '--cache_dir',
        type=str,
        default='./.cache',
        help=f"Cache directory for storing simulations"
    )
    exp_group.add_argument(
        '--load_index',
        type=bool,
        default=False,
        help=f"Loads cached index file for last experiment"
    )
    exp_group.add_argument(
        '--No_Observables',
        action='store_true',
        help="Disable observable downsampling defined in observables.tsv."
    )
    exp_group.add_argument(
        '--catchall',
        metavar='KEY=VALUE',
        nargs='*',
        help="Additional experiment arguments in key=value format."
    )
    exp_group.add_argument(
        '--run_all',
        default=None,
        help="Execute all benchmarks in a given directory."
    )

    return parser.parse_args()
