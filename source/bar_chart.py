"""Module for creating bar_chart animation."""
# TODO
# Refactor variables to init_anim() and animate()


import math
import operator
from statistics import geometric_mean
from typing import Callable, List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import Animation, FFMpegWriter
from matplotlib.artist import Artist
from matplotlib.axes._subplots import Axes
from matplotlib.backends.backend_qt5 import FigureManagerQT as FigureManager
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .tools import (comma_format_int, extract_csv, extract_pop_per_rep,
                    extract_priority, extract_priority_tuple, extract_reps,
                    extract_state_names, parse_states)
from .types import (BarContainer, CsvStateInfo, PlotBarsDict, PlotProps,
                    PlotTextDict, StateInfo, Text)


def format_plot_1(
        plt_1: Axes, x_vals: List[int], state_info_list: List[StateInfo]) -> PlotProps:
    """Adjust all properties of plot 1 to make it look nice.

    Add the x & y ticks, format those ticks, set the title, draw the mean
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

    y_formatter = FuncFormatter(comma_format_int())

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
    geo_mean_pop_per_seat: float = geometric_mean(pop_per_rep_list)

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
    """Adjust all properties of plot 2 to make it look nice.

    Add the x & y ticks, format those ticks, set the title, and place the
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
    """Adjust all properties of plot 3 to make it look nice.

    Add the x & y ticks, format those ticks, set the title, and place the
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
    """Adjust all properties of plot 4 to make it look nice.

    Add the x & y ticks, format those ticks, set the title, and place the
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
    """Adjust plot level properties (all subplots but not the entire window)."""
    plt.style.use("seaborn-dark")

    plt.subplots_adjust(top=0.955,
                        bottom=0.149,
                        left=0.067,
                        right=0.991,
                        hspace=0.553,
                        wspace=0.075)


def init_anim_factory(
        plt_bars_dict: PlotBarsDict, txt_dict: PlotTextDict, mean_line: Line2D) -> Callable:
    """Create an init_anim function that returns the needed artists.

    init_anim() doesn't allow for custom parameters to be passed. Therefore, we make them
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
    def init_anim() -> List[Artist]:
        """Return the initial artists on the plot.

        This is needed for the blitting algorithm to
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

        return [*bars, *txts, mean_line]
    return init_anim


def animate(
        frame: int, state_info_list: List[StateInfo],
        plt_bars_dict: PlotBarsDict, txt_dict: PlotTextDict, mean_line: Line2D) -> List[Artist]:
    """Calculate the new priority values and reps in each state.

    Called every frame of Matplotlib's `FuncAnimation`.This is passed the properties about
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
    # This adds the representative from the last frame's calculated max_pri
    for state_info in state_info_list:
        if state_info["max_pri"]:
            # print(f"{frame=} {state_info['name']=}")
            state_info["reps"] = state_info["reps"] + 1
            state_info["max_pri"] = False

    # This calculates the next state to give a rep in the next frame
    for state_info in state_info_list:
        state_info["priority"] = (
            state_info["pop"] *
            (1 / math.sqrt((state_info["reps"] + 1) * ((state_info["reps"] + 1) - 1))))
        state_info["pop_per_rep"] = state_info["pop"] / \
            state_info["reps"]

    max_index, _ = max(
        enumerate(state_info_list), key=extract_priority_tuple)

    # print(f"{frame=} {max_index=} {max_value=}")

    state_info_list[max_index]["max_pri"] = True

    update_plt1(plt_bars_dict["plt_1_bars"], state_info_list, mean_line,
                txt_dict, frame)
    update_plt2(plt_bars_dict["plt_2_bars"], state_info_list)
    update_plt3(plt_bars_dict["plt_3_bars"], state_info_list)

    bars: List[Rectangle] = [artist for container in list(
        plt_bars_dict.values()) for artist in container]
    txts: List[Text] = list(txt_dict.values())

    return [*bars, *txts, mean_line]


def update_plt1(
        plt_1_bars: BarContainer, state_info_list: List[StateInfo],
        mean_line: Line2D, txt_dict: PlotTextDict, frame: int) -> None:
    """Insert the new data on the plot.

    Re-plot all of the bars, move the mean line, and set the text of
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
    pop_per_rep_list = extract_pop_per_rep(state_info_list)

    mean_pop_per_seat: float = np.mean(pop_per_rep_list)
    std_dev_pop_per_seat: float = np.std(pop_per_rep_list)
    range_pop_per_seat: float = max(
        pop_per_rep_list) - min(pop_per_rep_list)
    geo_mean_pop_per_seat: float = geometric_mean(pop_per_rep_list)

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


def update_plt2(plt_2_bars: BarContainer,
                state_info_list: List[StateInfo]) -> None:
    """Insert the new data on the plot.

    Parameters
    ----------
    plt_2_bars : `BarContainer`
        The objects describing the plotted bars
    state_info_list : `List[StateInfo]`
        Continually updated list of state calculation info

    """
    for state, state_info in zip(plt_2_bars, state_info_list):
        state.set_height(state_info["reps"])


def update_plt3(plt_3_bars: BarContainer,
                state_info_list: List[StateInfo]) -> None:
    """Insert the new data on the plot.

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


def main(file: str, debug: bool) -> None:
    """Run all executable code."""
    rows: List[CsvStateInfo] = extract_csv(file)
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

    writer = FFMpegWriter(metadata=dict(title="ec-apportionment"))

    frames: int = 385
    # This doesn't work if FuncAnimation isn't assigned to a value,
    #  therefore, add disable-unused for `anim`
    anim: Animation = animation.FuncAnimation(  # pylint: disable=unused-variable
        fig, animate, fargs=(
            state_info_list, plt_bars_dict, txt_dict, mean_line),
        init_func=init_anim_factory(plt_bars_dict, txt_dict, mean_line),
        frames=frames, repeat=False, blit=True)

    fig_manager: Optional[FigureManager] = plt.get_current_fig_manager()
    if fig_manager is not None:
        fig_manager.set_window_title(
            "CGP Grey Electoral College speadsheet animated")

    if debug:
        plt.show()
    else:
        anim.save("recordings/ec-apportionment.mp4", writer=writer)
