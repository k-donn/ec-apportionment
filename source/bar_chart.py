"""
Show an animation of the Huntington–Hill apportionment method.

usage:
python3.7 source/bar_chart.py file
"""
# TODO
# Refactor out extract_x functions to curry function
# Refactor out magic number indices
# Add pictures preview in README

import csv
import math
import operator
from argparse import ArgumentParser
from typing import Callable, Dict, List, Tuple, TypedDict

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import Animation, FFMpegWriter
from matplotlib.artist import Artist
from matplotlib.axes._subplots import Axes
from matplotlib.backends.backend_qt5 import FigureManagerQT
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MultipleLocator


class StateInfo(TypedDict):
    """Dict with name, pop, reps, prio, pop_per_rep, and is max_pri"""
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
        reader = csv.reader(inp)
        res = list(reader)
    return res


def parse_states(raw_csv: List[CsvStateInfo]) -> List[StateInfo]:
    """Calculate the priority value, population per representative, and number
    of representatives for each state.

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


def comma_format_int() -> Callable:
    """Return a function that formats a number with a maginute comma. Future Me,
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


def calc_geo_mean(array: List[float]) -> float:
    """Calculate the geometric mean of an array of floats.

    Parameters
    ----------
    array : `List[float]`
        An array of float values

    Returns
    -------
    `float`
        The average of the products of the values

    """
    arr: np.ndarray = np.log(array)
    return np.exp(arr.sum()/len(arr))


def format_plot_1(
        plt_1: Axes, x_vals: List[int], state_info_list: List[StateInfo]) -> PlotProps:
    """Add the x & y ticks, format those ticks, set the title, draw the mean
    line, and place the text on the plot for the pop_per_rep plot.

    Parameters
    ----------
    plt_1 : `Axes`
        The object that describes the graph
    x_vals : `List[int]`
        The list of ints that shows the states' positions
    state_info_list : `List[StateInfo]`
        Continually updated list of state calculation info

    Returns
    -------
    `PlotProps`
        A tuple of the plotted bars, text, and line objects

    """
    state_names = extract_state_names(state_info_list)
    pop_per_rep_list = extract_pop_per_rep(state_info_list)

    plt_1_bars: BarContainer = plt_1.bar(x_vals, pop_per_rep_list,
                                         align="center")

    plt_1.set_xticks(x_vals)
    plt_1.set_xticklabels(state_names, rotation="vertical")

    y_formatter: FuncFormatter = FuncFormatter(comma_format_int())

    plt_1.set_ylabel("People/Representative")
    plt_1.set_yscale("log")
    plt_1.get_yaxis().set_major_formatter(y_formatter)

    plt_1.set_title("People per representative per state")

    plt_1.grid(axis="y", which="major", lw=2)
    plt_1.grid(axis="y", which="minor", lw=0.75)
    plt_1.grid(axis="x", lw=0.75)

    mean_pop_per_seat: float = np.mean(pop_per_rep_list)
    std_dev_pop_per_seat: float = np.std(pop_per_rep_list)
    range_pop_per_seat: float = max(
        pop_per_rep_list) - min(pop_per_rep_list)
    geo_mean_pop_per_seat: float = calc_geo_mean(pop_per_rep_list)

    res_dict: PlotTextDict = {}

    res_dict["seat_txt"] = plt_1.text(
        0.25, 0.75, f"Seat# 1", transform=plt_1.transAxes)
    res_dict["state_txt"] = plt_1.text(
        0.15, 0.85, "State: ", transform=plt_1.transAxes)
    res_dict["mean_txt"] = plt_1.text(
        0.45, 0.75, f"Mean: {mean_pop_per_seat:,.2f}", transform=plt_1.transAxes)
    res_dict["std_dev_txt"] = plt_1.text(
        0.35, 0.85, f"Std. Dev. {std_dev_pop_per_seat:,.2f}", transform=plt_1.transAxes)
    res_dict["range_txt"] = plt_1.text(
        0.70, 0.75, f"Range: {range_pop_per_seat:,.2f}", transform=plt_1.transAxes)
    res_dict["geo_mean_txt"] = plt_1.text(
        0.6, 0.85, f"Geo. Mean: {geo_mean_pop_per_seat:,.2f}", transform=plt_1.transAxes)
    mean_line: Line2D = plt_1.axhline(y=mean_pop_per_seat,
                                      xmin=0.0, xmax=1.0, color="r")

    return (plt_1_bars, mean_line, res_dict)


def format_plot_2(
        plt_2: Axes, x_vals: List[int], state_info_list: List[StateInfo]) -> BarContainer:
    """Add the x & y ticks, format those ticks, set the title, and place the
    text on the plot for the number of reps plot.

    Parameters
    ----------
    plt_2 : `Axes`
        The object that describes the graph
    x_vals : `List[int]`
        The list of ints that shows the states' position's
    state_info_list : `List[StateInfo]`
        Continually updated list of state calculation info

    Returns
    -------
    `BarContainer`
        The objects describing the plotted bars

    """
    state_names = extract_state_names(state_info_list)
    reps_list = extract_reps(state_info_list)

    plt_2_bars: BarContainer = plt_2.bar(
        x_vals, reps_list, align="center", color="r")
    plt_2.set_xticks(x_vals)
    plt_2.set_xticklabels(state_names, rotation="vertical")

    y_axis = plt_2.get_yaxis()

    minor_loc = MultipleLocator(5)
    y_axis.set_minor_locator(minor_loc)

    plt_2.set_ylabel("Representatives")
    plt_2.set_ylim(top=60, bottom=0)

    plt_2.grid(axis="y", which="major", lw=2)
    plt_2.grid(axis="y", which="minor", lw=0.75)
    plt_2.grid(axis="x", lw=0.75)

    plt_2.set_title("Representatives per state")

    return plt_2_bars


def format_plot_3(
        plt_3: Axes, x_vals: List[int], state_info_list: List[StateInfo]) -> BarContainer:
    """Add the x & y ticks, format those ticks, set the title, and place the
    text on the plot for the priority num plot.

    Parameters
    ----------
    plt_3 : `Axes`
        The object that describes the graph
    x_vals : `List[int]`
        The list of ints that shows the states' position's
    state_info_list : `List[StateInfo]`
        Continually updated list of state calculation info

    Returns
    -------
    `BarContainer`
        The objects describing the plotted bars

    """
    state_names = extract_state_names(state_info_list)
    priority_list = extract_priority(state_info_list)

    plt_3_bars: BarContainer = plt_3.bar(x_vals, priority_list,
                                         align="center", color="g")
    plt_3.set_xticks(x_vals)
    plt_3.set_xticklabels(state_names, rotation="vertical")

    y_formatter: FuncFormatter = FuncFormatter(comma_format_int())

    plt_3.set_ylabel("Priority value")
    plt_3.set_yscale("log")
    plt_3.get_yaxis().set_major_formatter(y_formatter)
    plt_3.text(0.3, 0.9, "Highlighted, is the state with the highest priority value",
               transform=plt_3.transAxes)

    plt_3.grid(axis="y", which="major", lw=2)
    plt_3.grid(axis="y", which="minor", lw=0.75)
    plt_3.grid(axis="x", lw=0.75)

    plt_3.set_title("Priority values per state")

    return plt_3_bars


def format_plot_4(plt_4: Axes) -> None:
    """Add the x & y ticks, format those ticks, set the title, and place the
    text on the plot for the empty text plot.

    Parameters
    ----------
    plt_4 : `Axes`
        The object that describes the graph

    """
    plt_4.text(0.5, 0.5, "CGP Grey Electoral College Spreadsheet graphed.",
               transform=plt_4.transAxes, fontsize=20, horizontalalignment="center")
    plt_4.axis("off")


def format_plt() -> None:
    """Adjust plot level properties (all subplots but not the entire window)"""
    plt.style.use("seaborn-dark")

    plt.subplots_adjust(top=0.955,
                        bottom=0.149,
                        left=0.067,
                        right=0.991,
                        hspace=0.553,
                        wspace=0.075)


def init_anim_factory(
        plt_bars_dict: PlotBarsDict, txt_dict: PlotTextDict, mean_line: Line2D) -> Callable:
    """Create an init_anim function that returns the needed artists. init_anim()
    doesn't allow for custom parameters to be passed. Therefore, we make them
    here.

    state_info_list : `List[StateInfo]`
        Continually updated list of state calculation info
    plt_bars_dict : `PlotBarsDict`
        A dictionary that links the name of each plot to its respective `BarContainer` instance
    txt_dict : `PlotTextDict`
        A dictionary that links the name of each text property to its `Text` object
    mean_line : `Line2D`
        The object describing the mean-line in the first plot

    Returns
    -------
    `Callable`
        The init_anim function that returns the initial artists on the plot

    """
    def init_anim() -> List:
        """Return the initial artists on the plot for the blitting algorithm to
        use.

        Returns
        -------
        `List[Artist]`
            All of the bars, texts, and the mean line on the plot.

        """

        # All the containers for each of the states in each plot
        plot_containers: List[BarContainer] = list(plt_bars_dict.values())
        bars: List[Rectangle] = [artist
                                 for container in plot_containers for artist in container]
        txts: List[Text] = list(txt_dict.values())

        return bars + txts + [mean_line]
    return init_anim


def animate(
        frame: int, state_info_list: List[StateInfo],
        plt_bars_dict: PlotBarsDict, txt_dict: PlotTextDict, mean_line: Line2D) -> List[Artist]:
    """Called every frame of Matplotlib's `FuncAnimation`. Calculate the new
    priority values and reps in each state. This is passed the properties about
    each of the subplots that we need to update and the previous frame's
    finished calculations. This makes calls to other functions that update each
    individual plot.

    Parameters
    ----------
    frame : `int`
        The current frame number
    state_info_list : `List[StateInfo]`
        Continually updated list of state calculation info
    plt_bars_dict : `PlotBarsDict`
        A dictionary that links the name of each plot to its respective `BarContainer` instance
    txt_dict : `PlotTextDict`
        A dictionary that links the name of each text property to its `Text` object
    mean_line : `Line2D`
        The object describing the mean-line in the first plot

    Returns
    -------
    `List[Artist]`
        All of the artists that the blitting algorithm needs to update

    """
    for state_info in state_info_list:
        if state_info["max_pri"]:
            state_info["reps"] = state_info["reps"] + 1
            state_info["max_pri"] = False

    for state_info in state_info_list:
        state_info["priority"] = (
            state_info["pop"] *
            (1 / math.sqrt((state_info["reps"] + 1) * ((state_info["reps"] + 1) - 1))))
        state_info["pop_per_rep"] = state_info["pop"] / \
            state_info["reps"]

    state_info_list[state_info_list.index(
        max(state_info_list, key=operator.itemgetter("priority")))]["max_pri"] = True

    update_plt1(plt_bars_dict["plt_1_bars"], state_info_list, mean_line,
                txt_dict, frame)
    update_plt2(plt_bars_dict["plt_2_bars"], state_info_list)
    update_plt3(plt_bars_dict["plt_3_bars"], state_info_list)

    bars: List[Rectangle] = [artist for container in list(
        plt_bars_dict.values()) for artist in container]
    txts: List[Text] = list(txt_dict.values())

    return bars + txts + [mean_line]


def update_plt1(
        plt_1_bars: BarContainer, state_info_list: List[StateInfo],
        mean_line: Line2D, txt_dict: PlotTextDict, frame: int) -> None:
    """Re-plot all of the bars, move the mean line, and set the text of
    everything on plot 1 with newly calculated data.

    Parameters
    ----------
    plt_1_bars : `BarContainer`
        The objects describing the plotted bars
    state_info_list : `List[StateInfo]`
        Continually updated list of state calculation info
    mean_line : `Line2D`
        The object describing the mean-line in the first plot
    txt_dict : `PlotTextDict`
        A dictionary that links the name of each text property to its `Text` object
    frame : `int`
        The current frame number

    """
    pop_per_rep_list = list(
        map(operator.itemgetter("pop_per_rep"), state_info_list))

    mean_pop_per_seat: float = np.mean(pop_per_rep_list)
    std_dev_pop_per_seat: float = np.std(pop_per_rep_list)
    range_pop_per_seat: float = max(
        pop_per_rep_list) - min(pop_per_rep_list)
    geo_mean_pop_per_seat: float = calc_geo_mean(pop_per_rep_list)

    max_state: str = max(
        state_info_list, key=operator.itemgetter("priority"))["name"]

    txt_dict["seat_txt"].set_text(
        f"Seat# {frame + 1}")
    txt_dict["state_txt"].set_text(
        f"State: {max_state}")
    txt_dict["mean_txt"].set_text(
        f"Mean: {mean_pop_per_seat:,.2f}")
    txt_dict["std_dev_txt"].set_text(
        f"Std. Dev. {std_dev_pop_per_seat:,.2f}")
    txt_dict["range_txt"].set_text(
        f"Range: {range_pop_per_seat:,.2f}")
    txt_dict["geo_mean_txt"].set_text(
        f"Geo. Mean: {geo_mean_pop_per_seat:,.2f}")

    mean_line.set_xdata([0, 1.0])
    mean_line.set_ydata([mean_pop_per_seat])

    for state, state_info in zip(plt_1_bars, state_info_list):
        state.set_height(state_info["pop_per_rep"])


def update_plt2(plt_2_bars: BarContainer, state_info_list: List[StateInfo]) -> None:
    """Re-plot all of the bars on plot 2 with newly calculated data.

    Parameters
    ----------
    plt_2_bars : `BarContainer`
        The objects describing the plotted bars
    state_info_list : `List[StateInfo]`
        Continually updated list of state calculation info

    """
    for state, state_info in zip(plt_2_bars, state_info_list):
        state.set_height(state_info["reps"])


def update_plt3(plt_3_bars: BarContainer, state_info_list: List[StateInfo]) -> None:
    """Re-plot all of the bars on plot 3 with newly calculated data.

    Parameters
    ----------
    plt_3_bars : `BarContainer`
        The objects describing the plotted bars
    state_info_list : `List[StateInfo]`
        Continually updated list of state calculation info

    """
    for state, state_info in zip(plt_3_bars, state_info_list):
        state.set_color("g")
        if state_info["max_pri"]:
            state.set_color("r")
        state.set_height(state_info["priority"])


def main() -> None:
    """Run all executable code."""

    parser: ArgumentParser = ArgumentParser(
        prog="python3.7 source/bar_chart.py",
        description="Show an animation of the Huntington–Hill apportionment method")
    parser.add_argument("-f", "--file", required=True,
                        help="Path to CSV state population data")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Show the plot instead of writing to file")

    args = parser.parse_args()

    rows: List[CsvStateInfo] = extract_csv(args.file)
    state_info_list: List[StateInfo] = parse_states(rows)

    fig: Figure = plt.figure(figsize=(16, 9), dpi=120)

    format_plt()

    plt_1: Axes = fig.add_subplot(221)
    plt_2: Axes = fig.add_subplot(222)
    plt_3: Axes = fig.add_subplot(223)
    plt_4: Axes = fig.add_subplot(224)

    x_pos: np.ndarray = np.arange(len(state_info_list))

    (plt_1_bars, mean_line, txt_dict) = format_plot_1(
        plt_1, x_pos, state_info_list)
    plt_2_bars: BarContainer = format_plot_2(plt_2, x_pos, state_info_list)
    plt_3_bars: BarContainer = format_plot_3(plt_3, x_pos, state_info_list)
    format_plot_4(plt_4)

    plt_bars_dict: PlotBarsDict = {"plt_1_bars": plt_1_bars,
                                   "plt_2_bars": plt_2_bars,
                                   "plt_3_bars": plt_3_bars}

    writer = FFMpegWriter(fps=5, bitrate=250000, extra_args=["-minrate", "650k", "-maxrate", "1M"],
                          metadata=dict(title="/u/ilikeplanes86"))

    frames: int = 385
    # This doesn't work if FuncAnimation isn't assigned to a value,
    #  therefore, add disable-unused for `anim`
    anim: Animation = animation.FuncAnimation(  # pylint: disable=unused-variable
        fig, animate, fargs=(
            state_info_list, plt_bars_dict, txt_dict, mean_line),
        init_func=init_anim_factory(plt_bars_dict, txt_dict, mean_line),
        frames=frames, repeat=False, blit=True)

    fig_manager: FigureManagerQT = plt.get_current_fig_manager()
    fig_manager.set_window_title(
        "CGP Grey Electoral College speadsheet animated")

    if args.debug:
        plt.show()
    else:
        anim.save("recordings/ec-apportionment.mp4", writer=writer)


if __name__ == "__main__":
    main()
