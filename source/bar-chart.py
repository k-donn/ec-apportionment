# TODO

import csv
import math
import operator
from argparse import ArgumentParser
from typing import Dict, Iterable, List, Tuple, Type, Union

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.animation import Animation
from matplotlib.axes._subplots import Axes
from matplotlib.backends.backend_qt5 import FigureManagerQT
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from scipy.stats.mstats import gmean

# dict with name, pop, priority values, pop_per_rep, and is max pri
StateInfo = Dict[str, Union[str, int, float, bool]]
# list containing name and pop
SimpleStateInfo = List[str]
# dict with all the plot's text objects
PlotTextDict = Dict[str, Text]
# attribute objects related to plot graphics
PlotProps = Tuple[BarContainer, Line2D, PlotTextDict]
# relates state names to respective bar objects
PlotBarsDict = Dict[str, BarContainer]


def extract_csv(fname: str) -> List[SimpleStateInfo]:
    """ 
    Turn the states in the csv file to a Python data structure.
    Parameters
    ----------
    fname: `str`
        Path to CSV state population data
    Returns
    -------
    `List[SimpleStateInfo]`
        The name and population of each state in the file
    """
    res: List[SimpleStateInfo]
    with open(fname) as inp:
        reader = csv.reader(inp)
        res = list(reader)
    return res


def parse_states(raw_csv: List[SimpleStateInfo]) -> List[StateInfo]:
    """
    Calculate the priority value, population per representative, and number of representatives for each 
    state.
    Parameters
    ----------
    raw_csv : `List[SimpleStateInfo]`
        The list of the population and name for each state


    Returns
    -------
    `List[StateInfo]`
        A list of the parsed attributes
    """
    max_priority = 0
    state_info_list = []
    for row in raw_csv:
        state_info = {}
        is_max = False
        state_info["name"] = row[0]
        state_info["pop"] = int(row[1])
        state_info["reps"] = 1
        state_info["pop_per_rep"] = state_info["pop"] / \
            state_info["reps"]
        fut_reps = state_info["reps"] + 1
        state_info["priority"] = state_info["pop"] * \
            (1 / math.sqrt(fut_reps * (fut_reps - 1)))
        if state_info["priority"] > max_priority:
            max_priority = state_info["priority"]
            is_max = True
        state_info["max_pri"] = is_max
        state_info_list.append(state_info)
    return state_info_list


def calc_geo_mean(array: List[float]) -> float:
    """
    Calculate the geometric mean of an array of floats.
    Parameters
    ----------
    array : `List[float]`
        An array of float values

    Returns
    -------
    `float`
        The average of the products of the values
    """
    a: np.ndarray = np.log(array)
    return np.exp(a.sum()/len(a))


def format_plot_1(plt_1: Axes, x_vals: List[int], pop_per_rep_list: List[float], state_names: List[str]) -> PlotProps:
    """
    Add the x & y ticks, format those ticks, set the title, draw the mean line, and place the text on the plot for
    the pop_per_rep plot.
    Parameters
    ----------
    plt_1 : `Axes`
        The object that describes the graph
    x_vals : `List[int]`
        The list of ints that shows the states' position's
    pop_per_rep_list : `List[float]`
        The list of population per representative values for all the states
    state_names : `List[str]`
        The list of state names

    Returns
    -------
    `PlotProps`
        A tuple of the plotted bars, text, and line objects
    """
    plt_1_bars: BarContainer = plt_1.bar(x_vals, pop_per_rep_list,
                                         align="center", alpha=0.5)
    plt_1.set_xticks(x_vals)
    plt_1.set_xticklabels(state_names, rotation=77)

    plt_1.set_ylabel("People/Representative")
    plt_1.set_yscale("log")
    plt_1.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt_1.set_title("Progression of people per representative in each state.")

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
        0.35, 0.85, f"Std. Dev. {std_dev_pop_per_seat}", transform=plt_1.transAxes)
    res_dict["range_txt"] = plt_1.text(
        0.70, 0.75, f"Range: {range_pop_per_seat}", transform=plt_1.transAxes)
    res_dict["geo_mean_txt"] = plt_1.text(
        0.6, 0.85, f"Geo. Mean: {geo_mean_pop_per_seat}", transform=plt_1.transAxes)
    mean_line: Line2D = plt_1.axhline(y=mean_pop_per_seat,
                                      xmin=0.0, xmax=1.0, color="r")

    plt_1.text(0.0, 0.0, "/u/ilikeplanes86", transform=plt_1.transAxes)

    return (plt_1_bars, mean_line, res_dict)


def format_plot_2(plt_2: Axes, x_vals: List[int], reps_list: List[int], state_names: List[str]) -> BarContainer:
    """
    Add the x & y ticks, format those ticks, set the title,  and place the text on the plot for
    the number of reps plot.
    Parameters
    ----------
    plt_2 : `Axes`
        The object that describes the graph
    x_vals : `List[int]`
        The list of ints that shows the states' position's
    reps_list : `List[int]`
        The list of the count of each states' representatives
    state_names : `List[str]`
        The list of state names

    Returns
    -------
    `BarContainer`
        The objects describing the plotted bars
    """
    plt_2_bars: BarContainer = plt_2.bar(
        x_vals, reps_list, align="center", alpha=0.5, color="r")
    plt_2.set_xticks(x_vals)
    plt_2.set_xticklabels(state_names, rotation=77)

    plt_2.set_ylabel("Representatives")
    plt_2.set_ylim(top=60, bottom=0)

    plt_2.set_title("Number of representatives in each state")

    return plt_2_bars


def format_plot_3(plt_3: Axes, x_vals: List[int], priority_list: List[float], state_names: List[str]) -> BarContainer:
    """
    Add the x & y ticks, format those ticks, set the title, and place the text on the plot for
    the priority num plot.
    Parameters
    ----------
    plt_3 : `Axes`
        The object that describes the graph
    x_vals : `List[int]`
        The list of ints that shows the states' position's
    priority_list : `List[float]`
        The list of each states' priority values
    state_names : `List[str]`
        The list of state names

    Returns
    -------
    `BarContainer`
        The objects describing the plotted bars
    """
    plt_3_bars: BarContainer = plt_3.bar(x_vals, priority_list,
                                         align="center", alpha=0.5, color="g")
    plt_3.set_xticks(x_vals)
    plt_3.set_xticklabels(state_names, rotation=77)

    plt_3.set_ylabel("Priority value")
    plt_3.set_yscale("log")
    plt_3.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt_3.text(0.3, 0.9, "Highlighted, is the state with the highest priority value",
               transform=plt_3.transAxes)

    plt_3.set_title("Priority values for each state")

    return plt_3_bars


def format_plot_4(plt_4: Axes) -> None:
    """
    Add the x & y ticks, format those ticks, set the title,and place the text on the plot for
    the empty text plot.
    Parameters
    ----------
    plt_4 : `Axes`
        The object that describes the graph

    """
    plt_4.text(0.5, 0.5, "CGP Grey Electoral College Spreadsheet graphed.",
               transform=plt_4.transAxes, fontsize=20, horizontalalignment="center")
    plt_4.axis("off")


def format_plt(plt: matplotlib.pyplot) -> None:
    """
    Adjust the size of the plot (all subplots but not the entire window)
    Parameters
    ----------
    plt : `matplotlib.pyplot`
        The matplotlib `pyplot` module

    """
    plt.subplots_adjust(top=0.963,
                        bottom=0.142,
                        left=0.064,
                        right=0.986,
                        hspace=0.495,
                        wspace=0.072)


def init_anim() -> None:
    """ 
    Called very first on Matplotlib's `FuncAnimation`.
    Nothing needs to be done. All initialization is done
    in the `format_plt_x` functions.
    """
    return


def animate(frame: int, state_info_list: List[StateInfo], plt_bars_dict: PlotBarsDict,  txt_dict: PlotTextDict, mean_line: Line2D) -> None:
    """
    Called every frame of Matplotlib's `FuncAnimation`. Calculate the 
    new priority values and reps in each state. This is passed the 
    properties about each of the subplots that we need to update and
    the previous frame's finished calculations. This makes calls to
    other functions that update each individual plot.
    Parameters
    ----------
    frame : `int`
        The current frame number
    state_info_list : `List[StateInfo]`
        The parsed attributes about each of the states (pop_per_rep, priority values, etc.)
    plt_bars_dict : `PlotBarsDict`
        A dictionary that links the name of each plot to its respective `BarContainer` instance
    txt_dict : `PlotTextDict`
        A dictionary that links the name of each text property to its `Text` object
    mean_line : `Line2D`
        The object describing the mean-line in the first plot
    """
    # print(f"Frame #{frame + 1}")
    for state_info in state_info_list:
        if state_info["max_pri"]:
            state_info["reps"] = state_info["reps"] + 1
            # print(f"Adding to {state_info['name']}")
            state_info["max_pri"] = False

    for state_info in state_info_list:
        state_info["priority"] = (state_info["pop"] *
                                  (1 / math.sqrt((state_info["reps"] + 1) * ((state_info["reps"] + 1) - 1))))
        state_info["pop_per_rep"] = state_info["pop"] / \
            state_info["reps"]

    state_info_list[state_info_list.index(
        max(state_info_list, key=lambda v: v["priority"]))]["max_pri"] = True

    update_plt1(plt_bars_dict["plt_1_bars"], state_info_list, mean_line,
                txt_dict, frame)
    update_plt2(plt_bars_dict["plt_2_bars"], state_info_list)
    update_plt3(plt_bars_dict["plt_3_bars"], state_info_list)


def update_plt1(plt_1_bars: BarContainer, state_info_list: List[StateInfo], mean_line: Line2D, txt_dict: PlotTextDict, frame: int) -> None:
    """
    Re-plot all of the bars, move the mean line, and set the text of everything on 
    plot 1 with newly calculated data.
    Parameters
    ----------
    plt_1_bars : `BarContainer`
        The objects describing the plotted bars
    state_info_list : `List[StateInfo]`
        The parsed attributes about each of the states (pop_per_rep, priority values, etc.)
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

    max_state: str = max(state_info_list, key=lambda v: v["priority"])["name"]

    txt_dict["seat_txt"].set_text(
        f"Seat# {frame + 1}")
    txt_dict["state_txt"].set_text(
        f"State: {max_state}")
    txt_dict["mean_txt"].set_text(
        f"Mean: {mean_pop_per_seat:,.2f}")
    txt_dict["std_dev_txt"].set_text(
        f"Std. Dev. {std_dev_pop_per_seat:,.2f}")
    txt_dict["range_txt"].set_text(
        f"Range: {range_pop_per_seat}")
    txt_dict["geo_mean_txt"].set_text(
        f"Geo. Mean: {geo_mean_pop_per_seat:,.2f}")

    mean_line.set_xdata([0, 1.0])
    mean_line.set_ydata([mean_pop_per_seat])

    for bar, state_info in zip(plt_1_bars, state_info_list):
        bar.set_height(state_info["pop_per_rep"])


def update_plt2(plt_2_bars: BarContainer, state_info_list: List[StateInfo]) -> None:
    """
    Re-plot all of the bars on plot 2 with newly calculated data.
    Parameters
    ----------
    plt_2_bars : `BarContainer`
        The objects describing the plotted bars
    state_info_list : `List[StateInfo]`
        The parsed attributes about each of the states (pop_per_rep, priority values, etc.)
    """
    for bar, state_info in zip(plt_2_bars, state_info_list):
        bar.set_height(state_info["reps"])


def update_plt3(plt_3_bars: BarContainer, state_info_list: List[StateInfo]) -> None:
    """
    Re-plot all of the bars on plot 3 with newly calculated data.
    Parameters
    ----------
    plt_3_bars : `BarContainer`
        The objects describing the plotted bars
    state_info_list : `List[StateInfo]`
        The parsed attributes about each of the states (pop_per_rep, priority values, etc.)
    """
    for bar, state_info in zip(plt_3_bars, state_info_list):
        bar.set_color("g")
        if state_info["max_pri"]:
            bar.set_color("r")
        bar.set_height(state_info["priority"])


def main() -> None:
    """ 
    Run all executable code
    """
    matplotlib.use("Qt5Agg")

    parser: ArgumentParser = ArgumentParser(prog="python3 source/bar-chart.py",
                                            description="Show an animation of the Huntington–Hill apportionment method")
    parser.add_argument("file", help="Path to CSV state population data")

    args = parser.parse_args()

    rows: List[SimpleStateInfo] = extract_csv(args.file)
    state_info_list: List[StateInfo] = parse_states(rows)

    state_names: List[str] = list(
        map(operator.itemgetter("name"), state_info_list))
    pop_per_rep_list: List[float] = list(
        map(operator.itemgetter("pop_per_rep"), state_info_list))
    reps_list: List[int] = list(
        map(operator.itemgetter("reps"), state_info_list))
    priority_list: List[float] = list(
        map(operator.itemgetter("priority"), state_info_list))

    fig: Figure = plt.figure()

    plt_1: Axes = fig.add_subplot(221)
    plt_2: Axes = fig.add_subplot(222)
    plt_3: Axes = fig.add_subplot(223)
    plt_4: Axes = fig.add_subplot(224)

    x_pos: np.ndarray = np.arange(len(state_info_list))

    (plt_1_bars, mean_line, txt_dict) = format_plot_1(
        plt_1, x_pos, pop_per_rep_list, state_names)
    plt_2_bars: BarContainer = format_plot_2(
        plt_2, x_pos, reps_list, state_names)
    plt_3_bars: BarContainer = format_plot_3(
        plt_3, x_pos, priority_list, state_names)
    format_plot_4(plt_4)

    format_plt(plt)

    plt_bars_dict: PlotBarsDict = {"plt_1_bars": plt_1_bars,
                                   "plt_2_bars": plt_2_bars,
                                   "plt_3_bars": plt_3_bars}

    frames: int = 385
    # This doesn't work if FuncAnimation isn't assigned to a value, hence, add disable-unused for `anim`
    anim: Animation = animation.FuncAnimation(  # pylint: disable=unused-variable
        fig, animate, fargs=(state_info_list, plt_bars_dict, txt_dict, mean_line), init_func=init_anim, frames=frames, interval=100, repeat=False)

    figManager: FigureManagerQT = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    figManager.set_window_title(
        "CGP Grey Electoral College speadsheet animated")

    plt.show()


if __name__ == "__main__":
    main()
