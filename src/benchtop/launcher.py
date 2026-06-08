"""CLI entry point: run one benchmark or all YAML files in a directory."""

import logging
import os
from typing import List

from types import SimpleNamespace

from benchtop.arguments import parse_args
from benchtop.experiment import Experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    args = parse_args()

    if args.command == "experiment":
        launch_experiment(args)
    else:
        print("No valid command provided. Use --help for guidance.")


def launch_experiment(args: SimpleNamespace) -> None:
    if args.run_all is not None:
        _run_all(args)
    else:
        assert args.path is not None, (
            "Provide a PEtab YAML path (-p) or use --run_all."
        )
        _run_experiment(args, args.path)


def _run_all(args: SimpleNamespace) -> None:
    for yaml_path in _get_list_of_experiments(args.run_all):
        assert os.path.exists(yaml_path), f"Experiment not found: {yaml_path}"
        _run_experiment(args, yaml_path)


def _run_experiment(args: SimpleNamespace, config_path: str) -> None:
    assert os.path.exists(config_path), f"Experiment not found: {config_path}"

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    experiment = Experiment(
        petab_yaml=config_path,
        cores=args.cores,
        cache_dir=args.cache_dir,
        load_index=args.load_index,
        verbose=args.verbose,
    )

    experiment.run(args.simulator, args)
    logger.debug("Simulation complete.")

    if args.No_Observables:
        logger.debug("Skipping observable calculation.")
    else:
        experiment.observable_calculation(args)
        logger.debug("Observable calculation complete.")


def _get_list_of_experiments(directory: str) -> List[str]:
    """Recursively find benchmark YAML files under directory."""
    yaml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".yaml", ".yml")):
                yaml_files.append(os.path.join(root, file))
    return yaml_files
