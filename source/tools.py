"""Methods used to perform operations on datatypes or other mundane ops."""
import csv
import math
from typing import Callable

from .types import CsvStateInfo, List, StateInfo, Tuple

# TODO


def parse_states(raw_csv: List[CsvStateInfo]) -> List[StateInfo]:
    """Construct the dict object for each state.

    Parameters
    ----------
    raw_csv : `List[SimpleStateInfo]`
        The list of the population and name for each state

    Returns
    -------
    `List[StateInfo]`
        A list of the parsed attributes

    """
    max_priority: float = 0
    state_info_list: List[StateInfo] = []
    for row in raw_csv:
        is_max = False
        name = row[0]
        pop = int(row[1])
        reps = 1
        pop_per_rep = pop / reps
        fut_reps = reps + 1
        priority = pop * (1 / math.sqrt(fut_reps * (fut_reps - 1)))

        if priority > max_priority:
            max_priority = priority
            is_max = True

        max_pri = is_max

        state_info: StateInfo = StateInfo(name=name, pop=pop, reps=reps,
                                          pop_per_rep=pop_per_rep,
                                          priority=priority, max_pri=max_pri)

        state_info_list.append(state_info)
    return state_info_list


def extract_csv(fname: str) -> List[CsvStateInfo]:
    """Turn the states in the csv file to a Python data structure.

    Parameters
    ----------
    fname: `str`
        Path to CSV state population data

    Returns
    -------
    `List[SimpleStateInfo]`
        The name and population of each state in the file

    """
    res: List[CsvStateInfo]
    with open(fname) as inp:
        parser = csv.reader(inp)
        res = list(parser)
    return res


def comma_format_int() -> Callable:
    """Return a function that inserts digits group seperators in a number.

    Future Me,
    I refactored this to a function because it has multiple references. Don't
    delete this.

    Returns
    -------
    `Callable`
        The formatting function

    """
    return lambda x, p: "{:,}".format(int(x))


def extract_state_names(state_info_list: List[StateInfo]) -> List[str]:
    """Extract the state_name propery from the list of state info.

    Parameters
    ----------
    state_info_list : List[StateInfo]
        Continually updated list of state calculation info

    Returns
    -------
    List[str]
        A list of all names
    """
    state_names: List[str] = [state["name"] for state in state_info_list]

    return state_names


def extract_pop_per_rep(state_info_list: List[StateInfo]) -> List[float]:
    """Extract the pop_per_rep propery from the list of state info.

    Parameters
    ----------
    state_info_list : List[StateInfo]
        Continually updated list of state calculation info

    Returns
    -------
    List[float]
        A list of all ratios
    """
    pop_per_rep_list: List[float] = [state["pop_per_rep"]
                                     for state in state_info_list]

    return pop_per_rep_list


def extract_reps(state_info_list: List[StateInfo]) -> List[int]:
    """Extract the priority propery from the list of state info.

    Parameters
    ----------
    state_info_list : List[StateInfo]
        Continually updated list of state calculation info

    Returns
    -------
    List[int]
        A list of all priority values
    """
    reps_list: List[int] = [state["reps"] for state in state_info_list]

    return reps_list


def extract_priority(state_info_list: List[StateInfo]) -> List[float]:
    """Extract the priority propery from the list of state info.

    Parameters
    ----------
    state_info_list : List[StateInfo]
        Continually updated list of state calculation info

    Returns
    -------
    List[float]
        A list of all priority values
    """
    priority_list: List[float] = [state["priority"]
                                  for state in state_info_list]

    return priority_list


def extract_priority_tuple(state_info_tuple: Tuple[int, StateInfo]) -> float:
    """Create a function to extract priority from the second part of an enumerated list.

    Example
    -------
    >>> l = [{'priority': 12}, {'priority': 24}]
    >>> e = enumerate(l)
    (1, {'priority': 12}), ...
    >>> max_index, max_dict = max(e, key=extract_priority_tuple)


    Parameters
    ----------
    state_info_tuple : Tuple[int, StateInfo]
        The generated Tuple of StateInfo

    Returns
    -------
    float
        The priority value of passed Tuple
    """
    return state_info_tuple[1]["priority"]
