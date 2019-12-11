# TODO
# Add infinite colors for tags

from typing import Type, Dict, List, Union, Tuple, Iterable
from matplotlib.axes._subplots import Axes
from matplotlib.figure import Figure
from matplotlib.container import BarContainer
from matplotlib.backends.backend_qt5 import FigureManagerQT
from matplotlib.text import Text
from matplotlib.animation import Animation
from matplotlib.lines import Line2D
from scipy.stats.mstats import gmean
import csv
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import matplotlib

# dict with name, pop, priority values, pop_per_rep, and is max pri
StateInfo = Dict[str, Union[str, int, float, bool]]
# list containing name and pop
SimpleStateInfo = List[str]
# dict with all the plot's text objects
PlotTextDict = Dict[str, Text]
# tuple with
# BarContainer of points
# mean line object
# dict with text objects
PlotProps = Tuple[BarContainer, Line2D, PlotTextDict]
# dict with plot name and associated plot bar objects
PlotBarsDict = Dict[str, BarContainer]


def extract_csv() -> List[SimpleStateInfo]:
    with open("state-populations.csv") as inp:
        reader = csv.reader(inp)
        return list(reader)


def parse_states(raw_csv: List[SimpleStateInfo]) -> List[StateInfo]:
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
    a: np.ndarray = np.log(array)
    return np.exp(a.sum()/len(a))


# bar chart of pop_per_rep
def format_plot_1(plt_1: Axes, x_vals: List[int], pop_per_rep_list: List[float], state_names: List[str]) -> PlotProps:
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

    seat_txt: Text = plt_1.text(
        0.25, 0.75, f"Seat# 1", transform=plt_1.transAxes)
    state_txt: Text = plt_1.text(
        0.15, 0.85, "State: ", transform=plt_1.transAxes)
    mean_txt: Text = plt_1.text(
        0.45, 0.75, f"Mean: {mean_pop_per_seat:,.2f}", transform=plt_1.transAxes)
    std_dev_txt: Text = plt_1.text(
        0.35, 0.85, f"Std. Dev. {std_dev_pop_per_seat}", transform=plt_1.transAxes)
    range_txt: Text = plt_1.text(
        0.70, 0.75, f"Range: {range_pop_per_seat}", transform=plt_1.transAxes)
    geo_mean_txt: Text = plt_1.text(
        0.6, 0.85, f"Geo. Mean: {geo_mean_pop_per_seat}", transform=plt_1.transAxes)
    mean_line: Line2D = plt_1.axhline(y=mean_pop_per_seat,
                                      xmin=0.0, xmax=1.0, color="r")

    plt_1.text(0.0, 0.0, "/u/ilikeplanes86", transform=plt_1.transAxes)

    return (plt_1_bars, mean_line, {"seat_txt": seat_txt, "state_txt": state_txt, "mean_txt": mean_txt, "std_dev_txt": std_dev_txt, "range_txt": range_txt, "geo_mean_txt": geo_mean_txt})


# bar chart of number of reps
def format_plot_2(plt_2: Axes, x_vals: List[int], reps_list: List[int], state_names: List[str]) -> BarContainer:
    plt_2_bars: BarContainer = plt_2.bar(
        x_vals, reps_list, align="center", alpha=0.5, color="r")
    plt_2.set_xticks(x_vals)
    plt_2.set_xticklabels(state_names, rotation=77)

    plt_2.set_ylabel("Representatives")
    plt_2.set_ylim(60)
    # Y-axis gets flpped for some reason
    plt_2.invert_yaxis()

    plt_2.set_title("Number of representatives in each state")

    return plt_2_bars


# bar chart of priority nums
def format_plot_3(plt_3: Axes, x_vals: List[int], priority_list: List[float], state_names: List[str]) -> BarContainer:
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
    plt_4.text(0.5, 0.5, "CGP Grey Electoral College Spreadsheet graphed.",
               transform=plt_4.transAxes, fontsize=20, horizontalalignment="center")
    plt_4.axis("off")


def format_plt(plt: matplotlib.pyplot) -> None:
    plt.subplots_adjust(top=0.963,
                        bottom=0.142,
                        left=0.064,
                        right=0.986,
                        hspace=0.495,
                        wspace=0.072)


def init_anim() -> None:
    return


def animate(frame: int, state_info_list: List[StateInfo], plt_bars_dict: PlotBarsDict,  txt_dict: PlotTextDict, mean_line: Line2D) -> None:
    print(f"Frame #{frame + 1}")
    for state_info in state_info_list:
        if state_info["max_pri"]:
            state_info["reps"] = state_info["reps"] + 1
            print(f"Adding to {state_info['name']}")
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


def update_plt2(plt_2_bars: BarContainer, state_info_list: List[StateInfo]):
    for bar, state_info in zip(plt_2_bars, state_info_list):
        bar.set_height(state_info["reps"])


def update_plt3(plt_3_bars: BarContainer, state_info_list: List[StateInfo]):
    for bar, state_info in zip(plt_3_bars, state_info_list):
        bar.set_color("g")
        if state_info["max_pri"]:
            bar.set_color("r")
        bar.set_height(state_info["priority"])


def main():
    matplotlib.use("Qt5Agg")

    rows: List[SimpleStateInfo] = extract_csv()
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
    plt_2_bars: [BarContainer] = format_plot_2(
        plt_2, x_pos, reps_list, state_names)
    plt_3_bars: [BarContainer] = format_plot_3(
        plt_3, x_pos, priority_list, state_names)
    format_plot_4(plt_4)

    format_plt(plt)

    plt_bars_dict = {"plt_1_bars": plt_1_bars,
                     "plt_2_bars": plt_2_bars,
                     "plt_3_bars": plt_3_bars}

    # account for frame zero
    frames: int = 385
    animation.FuncAnimation(
        fig, animate, fargs=(state_info_list, plt_bars_dict, txt_dict, mean_line), init_func=init_anim, frames=frames, interval=100, repeat=False)

    figManager: FigureManagerQT = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    figManager.set_window_title(
        "CGP Grey Electoral College speadsheet animated")

    plt.show()
    # anim.save("bar-chart-autorecord.mp4", writer=writer)


if __name__ == "__main__":
    main()
