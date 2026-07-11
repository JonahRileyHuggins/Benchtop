"""Rover hybrid (BNGsim + StochMod) simulator wrapper."""

import logging

import pandas as pd
from rover import HybridSimulator

from benchtop._abstract_simulator import AbstractSimulator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RoverSimulator(AbstractSimulator):
    """Load partitioned SBML via Rover; couple BNGsim ODEs with StochMod tau-leap.

    Expects ``args.model_paths`` to list two SBML files in order:
    ``[deterministic_interactions.xml, stochastic_gene_expression.xml]``.

    Condition overrides for species are interpreted as molecule counts (Rover's
    shared-store currency). Parameter overrides are routed to BNGsim and/or
    StochMod by id.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def load(self, args, **kwargs):
        paths = getattr(args, "model_paths", None)
        if not paths or len(paths) < 2:
            raise ValueError(
                "Rover requires two SBML files in model_paths: "
                "[deterministic, stochastic]"
            )

        deterministic_sbml = paths[0]
        stochastic_sbml = paths[1]
        coupling_dt = float(kwargs.get("dt", kwargs.get("coupling_dt", 30.0)))
        bngsim_kwargs = kwargs.get("bngsim_kwargs", {"codegen": True})

        logger.debug(
            "Loading Rover hybrid: det=%s stoch=%s dt=%s",
            deterministic_sbml,
            stochastic_sbml,
            coupling_dt,
        )
        self.tool = HybridSimulator(
            deterministic_sbml,
            stochastic_sbml,
            dt=coupling_dt,
            bngsim_kwargs=bngsim_kwargs,
            initial_counts=kwargs.get("initial_counts"),
        )

    def simulate(self, start, stop, step) -> pd.DataFrame:
        t0 = float(start)
        t1 = float(stop + step)
        dt = float(step)

        logger.debug("Rover simulate t_span=(%s, %s) dt=%s", t0, t1, dt)
        self.tool.run(t_span=(t0, t1), dt=dt)
        return self.tool.to_dataframe()

    def modify(self, component: str, value: int | float) -> None:
        logger.debug("Setting %s = %s", component, value)
        try:
            self.tool.update(component, float(value))
        except KeyError as e:
            raise ValueError(
                f"Component '{component}' not found in Rover model species or parameters"
            ) from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error setting {component}: {e}") from e
