import os
import sys
import random

from wrappers.sparced_wrapper import WrapSPARCED

sys.path.append('bin/')

args = ("./SPARCED.xml", "./SPARCED/", 0)

# what should the abstract class do as a default?
# load the simulator, for one. 
# it has 4 methods, should probably check all
# Lets test load first, then modify

def test_abstractsim_modify() -> None:
    """Verify model settings are reflected in stored model"""

    # initialize object
    sparced = WrapSPARCED(args)

    # find number of species in SPARCED, 
    # species_initializations is a WrapSPARCED child class specific member
    sp_len = len(sparced.tool.species_initializations)

    pos = random.randint(1, sp_len)

    val = random.random()