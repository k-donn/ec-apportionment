"""All types present in ec-apportionment code."""
from typing import Dict, List, Tuple, TypedDict

from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from matplotlib.text import Text

# TODO
# - Use TypedDict on other types to whitelist proper keys


class StateInfo(TypedDict):
    """Dict with name, pop, reps, prio, pop_per_rep, and is max_pri."""

    name: str
    pop: int
    reps: int
    pop_per_rep: float
    priority: float
    max_pri: bool


# list containing name and pop
CsvStateInfo = List[str]
# dict with all the plot's text objects
PlotTextDict = Dict[str, Text]
# attribute objects related to plot graphics
PlotProps = Tuple[BarContainer, Line2D, PlotTextDict]
# relates state names to respective bar objects
PlotBarsDict = Dict[str, BarContainer]
