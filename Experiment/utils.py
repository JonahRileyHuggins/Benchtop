#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script name: petab_yamlpy
date: 11 -07 -2025
Author: Jonah R. Huggins

Description: shared-utility functions for Experiments class.

"""
# -----------------------Package Import & Defined Arguements-------------------#
import os, sys
from pathlib import Path
import textwrap

def get_project_root() -> Path:
    return os.path.join(Path.home(), ".local", "share", "SingleCell")


@staticmethod
def identifier_generator():
    """This function generates a unique identifier for the iterative
        of each simulation process.
    output:
        returns the unique identifier
    """
    import uuid

    identifier = str(uuid.uuid4())

    del uuid

    return identifier

@staticmethod
def tasks_this_round(size, total_jobs, round_number):
    """Calculate the number of tasks for the current round
    input:
        size: int - the total number of processes assigned
        total_jobs: int - the total number of tasks

    output:
        returns the number of tasks for the current round
    """
    number_of_rounds = -(-total_jobs // size)

    tasks_per_round = size
    remainder = total_jobs % size

    # This accounts for pythonic indexing starting at 0
    round_number += 1

    if round_number < number_of_rounds:
        tasks_this_round = tasks_per_round
    elif round_number == number_of_rounds and remainder != 0:
        tasks_this_round = remainder
    elif round_number == number_of_rounds and remainder == 0:
        tasks_this_round = tasks_per_round
    else:
        # provide an error and message exit
        raise ValueError("Round number exceeds the number of rounds")

    return tasks_this_round

@staticmethod 
def parse_kwargs(arg_list: list)-> dict:
    """Parses catchall function."""


    kwargs = {}


    for arg in arg_list:
        if '=' not in arg:
            raise ValueError(f"Invalid argument format: {arg}. Use key=value.")
        else:
            key, value = arg.split('=', 1)
            kwargs[key] = value


    return kwargs

def add_pysinglecell_path():
    # 1) env var
    env = os.getenv("SINGLECELL_PATH")
    print('hallejluah there was a path var found!')
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            sys.path.insert(0, str(p) + "/build")
            return True

    # 2) config file
    cfg = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".local")) / "share" / "SingleCell" / "build"
    if os.path.exists(cfg):
        for f in cfg.glob("pySingleCell*.so"):
            if f.is_file():
                sys.path.insert(0, str(cfg.resolve()))
                return env
        sys.path.insert(0, str(cfg))
    # 3) fallback: repo-relative build (only for dev)
    try:
        repo_root = Path(__file__).resolve().parents[4]  # adjust as necessary
        candidate = repo_root / "build"
        for f in candidate.glob("pySingleCell*.so"):
            if f.is_file():
                sys.path.insert(0, str(candidate.resolve()))
                return True
    except Exception:
        pass

    return False

def get_pysinglecell() -> os.PathLike | str: 
    if not add_pysinglecell_path():
        raise ModuleNotFoundError(textwrap.dedent("""\
            Could not find pySingleCell. Fix by:
            * setting SINGLECELL_PATH to the build dir: export SINGLECELL_PATH=/abs/path/to/build
            * or creating ~/.config/singlecell/pysinglecell_path containing that path
            * or packaging the .so into the wheel so pipx install includes it
        """))
    env = os.getenv("SINGLECELL_PATH")
    print('path was appended. Print statement for error stuff')
    return env
