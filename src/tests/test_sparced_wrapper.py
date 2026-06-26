import random

import pytest

pytest.importorskip("amici")

from wrappers.sparced_wrapper import WrapSPARCED

args = ("./SPARCED.xml", "./SPARCED/", 0)


def test_abstractsim_modify() -> None:
    """Verify model settings are reflected in stored model."""
    sparced = WrapSPARCED(args)

    sp_len = len(sparced.tool.species_initializations)
    pos = random.randint(1, sp_len)
    val = random.random()
