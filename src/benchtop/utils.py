#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script name: petab_yamlpy
date: 11 -07 -2025
Author: Jonah R. Huggins

Description: shared-utility functions for Experiments class.

"""
# -----------------------Package Import & Defined Arguements-------------------#

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
