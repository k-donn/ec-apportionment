"""Methods used to perform operations on datatypes or other mundane ops."""
import csv
import operator
from typing import Callable

from .types import CsvStateInfo, StateInfo, List, Tuple

# TODO
# - Use more pythonic code versus map


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
    state_names: List[str] = list(
        map(operator.itemgetter("name"), state_info_list))

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
    pop_per_rep_list: List[float] = list(
        map(operator.itemgetter("pop_per_rep"), state_info_list))

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
    reps_list: List[int] = list(
        map(operator.itemgetter("reps"), state_info_list))

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
    priority_list: List[float] = list(
        map(operator.itemgetter("priority"), state_info_list))

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
