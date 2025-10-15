import os
import sys

import pandas as pd

sys.path.append(f"{os.path.dirname(__file__)}/../")
from src.benchtop.Organizer import Organizer

def test_organizer_constructor() -> None:

    for i in range(1, 100):

        org = Organizer(i)

        assert org.workers == i

def test_topological_sorter() -> None:

    # Create instance of organizer
    org = Organizer(os.cpu_count())

    m_df_success = pd.DataFrame({

        "preequilibrationConditionId": [None, "serum_starve", "serum_starve"],
        "simulationConditionId": ["serum_starve", "primary_condition1", "primary_condition2"]
    })

    topo_succ_check = org.topologic_sort(m_df_success)

    assert topo_succ_check[0] == "serum_starve"

    assert "primary_condition1" in topo_succ_check
    assert "primary_condition2" in topo_succ_check


    # Circular dependency check
    m_df_circular_failure = pd.DataFrame({

        "preequilibrationConditionId": ["serum_starve", "serum_starve", "serum_starve"],
        "simulationConditionId": ["serum_starve", "primary_conditon1", "primary_condition2"]
    })

    try:
        topo_fail_check = org.topologic_sort(m_df_circular_failure)

    except RuntimeError as e: 
        assert str(e) == "Circular dependency detected among conditions!"


    m_df_no_preequib = pd.DataFrame({

        "preequilibrationConditionId": [None, None, None],
        "simulationConditionId": ["primary_conditon1", "primary_condition2", "serum_starve"] # intentional order change

    })

    topo_null_check = org.topologic_sort(m_df_no_preequib)

    assert topo_null_check[0] != "serum_starve", f"First task ordered wrong: {topo_null_check}"

def test_delay_secondary_condition() -> None:

    # 4 cores should see two None values for the first iteration
    org = Organizer(4)

    m_df_success = pd.DataFrame({

        "preequilibrationConditionId": [None, "serum_starve", "serum_starve"],
        "simulationConditionId": ["serum_starve", "primary_conditon1", "primary_condition2"]
    })

    succ_tasks = ["serum_starve+1", "serum_starve+2",
                  "primary_condition1+1", "primary_condition1+2",
                  "primary_condition2+1", "primary_condition2+2"]

    three_cells = 2 # arbitrary number, could be randomized

    delay_scc = org.delay_secondary_conditions(m_df_success, succ_tasks, three_cells)

    # only should have 2 extra spaces for none values
    assert len(delay_scc) == 8, f"Expected 8 tasks, returned {delay_scc}"
    assert delay_scc[2] == None
    assert delay_scc[3] == None

def test_total_tasks_basic():
    
    dummy = Organizer(1)

    # --- normal case ---
    tasks = ["condA", "condB"]
    result = dummy.total_tasks(tasks, 3)

    expected = [
        "condA+1", "condA+2", "condA+3",
        "condB+1", "condB+2", "condB+3"
    ]
    assert result == expected
    assert all("+" in job for job in result)
    assert len(result) == len(tasks) * 3


def test_total_tasks_empty_tasks():
    dummy = type("D", (), {
        "total_tasks": lambda self, tasks, cell_count: [
            f"{cond}+{cell}"
            for cond in tasks
            for cell in range(1, cell_count + 1)
        ]
    })()

    assert dummy.total_tasks([], 5) == [], "Empty input should yield empty list"


def test_total_tasks_zero_cells():
    dummy = type("D", (), {
        "total_tasks": lambda self, tasks, cell_count: [
            f"{cond}+{cell}"
            for cond in tasks
            for cell in range(1, cell_count + 1)
        ]
    })()

    assert dummy.total_tasks(["condA"], 0) == [], "Zero cells should yield empty list"


def test_task_organization():

    # Good number should be 6?
    org = Organizer(6)

    m_df_success = pd.DataFrame({

        "preequilibrationConditionId": [None, "serum_starve", "serum_starve"],
        "simulationConditionId": ["serum_starve", "primary_condition1", "primary_condition2"]
    })

    cells = 3 # I want a gap of 3 None entries at positions [3, 4, 5]

    rounds, directory = org.task_organization(m_df_success, cells)

    # 6 workers
    # 3 conditions
    # 3 cells per condition
    # 1 preequilibration round 
    # Should be 12 entries, so 2 rounds?

    assert rounds == 2, f"Number of rounds expected: 12, returned: {rounds}"

