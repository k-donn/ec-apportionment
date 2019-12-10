# TODO
# Move plotting into funcs

from typing import Type, Dict, List
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


def extract_csv():
    with open("state-populations.csv") as inp:
        reader = csv.reader(inp)
        return list(reader)


def parse_states(raw_csv):
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


matplotlib.use("Qt5Agg")


def calc_geo_mean(iterable):
    a = np.log(iterable)
    return np.exp(a.sum()/len(a))


# seat_txt: Type[Text] = plt_1.text(
#     0.25, 0.75, f"Seat# 1", transform=plt_1.transAxes)
# state_txt: Type[Text] = plt_1.text(
#     0.15, 0.85, "State: ", transform=plt_1.transAxes)
# mean_txt: Type[Text] = plt_1.text(
#     0.45, 0.75, f"Mean: {mean_people_per_seat:,.2f}", transform=plt_1.transAxes)
# std_dev_txt: Type[Text] = plt_1.text(
#     0.35, 0.85, f"Std. Dev. {std_dev_people_per_seat}", transform=plt_1.transAxes)
# range_txt: Type[Text] = plt_1.text(
#     0.70, 0.75, f"Range: {range_people_per_seat}", transform=plt_1.transAxes)
# geo_mean_txt: Type[Text] = plt_1.text(
#     0.6, 0.85, f"Geo. Mean: {geo_mean_people_per_seat}", transform=plt_1.transAxes)
# mean_line: Type[Line2D] = plt_1.axhline(y=mean_people_per_seat,
#                                         xmin=0.0, xmax=1.0, color="r")


# bar chart of pop_per_rep
def format_plot_1(plt_1, x_vals, y_vals, state_names):
    plt_1_bars: Type[BarContainer] = plt_1.bar(x_vals, y_vals,
                                               align="center", alpha=0.5)
    plt_1.set_xticks(x_vals)
    plt_1.set_xticklabels(state_names, rotation=77)

    plt_1.set_ylabel("People/Representative")
    plt_1.set_yscale("log")
    plt_1.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt_1.set_title("Progression of people per representative in each state.")

    plt_1.text(0.0, 0.0, "/u/ilikeplanes86", transform=plt_1.transAxes)

    return plt_1_bars


# bar chart of number of reps
def format_plot_2(plt_2, x_vals, y_vals, state_names):
    plt_2_bars: Type[BarContainer] = plt_2.bar(
        x_vals, y_vals, align="center", alpha=0.5, color="r")
    plt_2.set_xticks(x_vals)
    plt_2.set_xticklabels(state_names, rotation=77)

    plt_2.set_ylabel("Representatives")
    plt_2.set_ylim(60)
    # Y-axis gets flpped for some reason
    plt_2.invert_yaxis()

    plt_2.set_title("Number of representatives in each state")

    return plt_2_bars


# bar chart of priority nums
def format_plot_3(plt_3, x_vals, y_vals, state_names):
    plt_3_bars: Type[BarContainer] = plt_3.bar(x_vals, y_vals,
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


def format_plot_4(plt_4):
    plt_4.text(0.5, 0.5, "CGP Grey Electoral College Spreadsheet graphed.",
               transform=plt_4.transAxes, fontsize=20, horizontalalignment="center")


def format_plt(plt):
    plt.subplots_adjust(top=0.964,
                        bottom=0.138,
                        left=0.064,
                        right=0.986,
                        hspace=0.456,
                        wspace=0.072)


def animate(frame: int) -> None:
    if frame < 2:
        return


if __name__ == "__main__":
    rows = extract_csv()
    state_info = parse_states(rows)

    max_state: str = max(state_info, key=lambda v: v["priority"])["name"]

    pop_per_rep_list = list(
        map(operator.itemgetter("pop_per_rep"), state_info))
    state_names = list(map(operator.itemgetter("name"), state_info))
    reps_list = list(map(operator.itemgetter("reps"), state_info))
    priority_list = list(map(operator.itemgetter("priority"), state_info))

    mean_people_per_seat: float = np.mean(pop_per_rep_list)
    std_dev_people_per_seat: float = np.std(pop_per_rep_list)
    range_people_per_seat: float = max(
        pop_per_rep_list) - min(pop_per_rep_list)
    geo_mean_people_per_seat: float = calc_geo_mean(pop_per_rep_list)

    fig: Type[Figure] = plt.figure()

    plt_1: Type[Axes] = fig.add_subplot(221)
    plt_2: Type[Axes] = fig.add_subplot(222)
    plt_3: Type[Axes] = fig.add_subplot(223)
    plt_4: Type[Axes] = fig.add_subplot(224)

    y_pos = np.arange(len(state_info))
    x_pos = np.arange(len(state_info))

    format_plot_1(plt_1, x_pos, pop_per_rep_list, state_names)
    format_plot_2(plt_2, x_pos, reps_list, state_names)
    format_plot_3(plt_3, x_pos, priority_list, state_names)
    format_plot_4(plt_4)

    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='/u/ilikeplanes86'), bitrate=1800)

    # account for frame zero
    frames: int = 386
    # anim: Animation = animation.FuncAnimation(
    #     fig, animate, repeat=False, blit=False, frames=frames, interval=10)

    figManager: Type[FigureManagerQT] = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    figManager.set_window_title(
        "CGP Grey Electoral College speadsheet animated")

    plt.show()
    # anim.save("bar-chart-autorecord.mp4", writer=writer)
