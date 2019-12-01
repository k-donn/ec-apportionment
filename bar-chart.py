# TODO
#

from typing import Type, Dict, List
from matplotlib.axes._subplots import Axes
from matplotlib.figure import Figure
from matplotlib.container import BarContainer
from matplotlib.backends.backend_qt5 import FigureManagerQT
from matplotlib.text import Text
from matplotlib.animation import Animation
from matplotlib.lines import Line2D
from scipy.stats.mstats import gmean
import math
import csv
import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import matplotlib

matplotlib.use("Qt5Agg")

with open("state-populations.csv") as inp:
    reader = csv.reader(inp)
    state_pops_name: Type[Dict[str, int]] = {
        rows[0]: int(rows[1]) for rows in reader}
    state_names: List[str] = list(state_pops_name.keys())
    state_pops: List[int] = list(state_pops_name.values())

state_reps: List[int] = [1] * 50
state_reps_name: Type[Dict[str, int]] = dict(zip(state_names, state_reps))


def calc_state_people_per_seat(state_pops: List[float], state_reps: List[int]) -> List[float]:
    return [pop / reps for pop,
            reps in dict(zip(state_pops, state_reps)).items()]


def calc_priority_nums(state_names: List[str], state_reps_name: Type[Dict[str, int]], state_pops_name: Type[Dict[str, int]]) -> List[float]:
    res: List[float] = []
    for state in state_names:
        fut_state_reps: int = state_reps_name[state] + 1
        res.append(
            state_pops_name[state] * (1 / math.sqrt(fut_state_reps * (fut_state_reps - 1))))
    return res


def calc_geo_mean(iterable):
    a = np.log(iterable)
    return np.exp(a.sum()/len(a))


state_people_per_seat: List[float] = []
state_people_per_seat = calc_state_people_per_seat(state_pops, state_reps)

max_state: str = ""
mean_people_per_seat: float = np.mean(state_people_per_seat)
std_dev_people_per_seat: float = np.std(state_people_per_seat)
range_people_per_seat: float = 0
geo_mean_people_per_seat: float = calc_geo_mean(state_people_per_seat)
state_priority_nums: List[float] = calc_priority_nums(
    state_names, state_reps_name, state_pops_name)

y_pos = np.arange(len(state_names))
x_pos = np.arange(len(state_names))

fig: Type[Figure] = plt.figure()

plt_1: Type[Axes] = fig.add_subplot(221)
plt_2: Type[Axes] = fig.add_subplot(222)
plt_3: Type[Axes] = fig.add_subplot(223)
plt_4: Type[Axes] = fig.add_subplot(224)
plt_4.text(0.5, 0.5, "CGP Grey Electoral College Spreadsheet graphed.",
           transform=plt_4.transAxes, fontsize=20, horizontalalignment="center")

plt_1.text(0.0, 0.0, "/u/ilikeplanes86", transform=plt_1.transAxes)
seat_txt: Type[Text] = plt_1.text(
    0.25, 0.75, f"Seat# 1", transform=plt_1.transAxes)
state_txt: Type[Text] = plt_1.text(
    0.15, 0.85, "State: ", transform=plt_1.transAxes)
mean_txt: Type[Text] = plt_1.text(
    0.45, 0.75, f"Mean: {mean_people_per_seat:,.2f}", transform=plt_1.transAxes)
std_dev_txt: Type[Text] = plt_1.text(
    0.35, 0.85, f"Std. Dev. {std_dev_people_per_seat}", transform=plt_1.transAxes)
range_txt: Type[Text] = plt_1.text(
    0.70, 0.75, f"Range: {range_people_per_seat}", transform=plt_1.transAxes)
geo_mean_txt: Type[Text] = plt_1.text(
    0.6, 0.85, f"Geo. Mean: {geo_mean_people_per_seat}", transform=plt_1.transAxes)
mean_line: Type[Line2D] = plt_1.axhline(y=mean_people_per_seat,
                                        xmin=0.0, xmax=1.0, color="r")


plt_1_bars: Type[BarContainer] = plt_1.bar(y_pos, state_people_per_seat,
                                           align="center", alpha=0.5)
plt_1.set_xticks(x_pos)
plt_1.set_xticklabels(state_names, rotation=77)

plt_1.set_ylabel("People/Representative")
plt_1.set_yscale("log")
plt_1.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt_1.set_title("Progression of people per representative in each state.")

# bar chart of number of reps
plt_2_bars: Type[BarContainer] = plt_2.bar(
    y_pos, state_reps, align="center", alpha=0.5, color="r")
plt_2.set_xticks(x_pos)
plt_2.set_xticklabels(state_names, rotation=77)

plt_2.set_ylabel("Representatives")
plt_2.set_ylim(60)
# Y-axis gets flpped for some reason
plt_2.invert_yaxis()

plt_2.set_title("Number of representatives in each state")

plt_3_bars: Type[BarContainer] = plt_3.bar(y_pos, state_priority_nums,
                                           align="center", alpha=0.5, color="g")
plt_3.set_xticks(x_pos)
plt_3.set_xticklabels(state_names, rotation=77)

plt_3.set_ylabel("Priority value")
plt_3.set_yscale("log")
plt_3.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt_3.text(0.3, 0.9, "Highlighted, is the state with the highest priority value",
           transform=plt_3.transAxes)

plt_3.set_title("Priority values for each state")

plt.subplots_adjust(top=0.964,
                    bottom=0.138,
                    left=0.064,
                    right=0.986,
                    hspace=0.456,
                    wspace=0.072)


def animate(frame: int) -> None:
    if frame < 2:
        return

    # Plot 1
    state_priority_nums = calc_priority_nums(
        state_names, state_reps_name, state_pops_name)
    state_priority_name = dict(zip(state_names, state_priority_nums))

    max_state = max(state_priority_name.items(),
                    key=operator.itemgetter(1))[0]

    state_reps_name[max_state] = state_reps_name[max_state] + 1

    state_people_per_seat = []
    state_people_per_seat = calc_state_people_per_seat(
        state_pops, list(state_reps_name.values()))
    state_people_per_seat_name = dict(zip(state_names, state_people_per_seat))

    mean_people_per_seat = np.mean(state_people_per_seat)
    std_dev_people_per_seat = np.std(state_people_per_seat)
    range_people_per_seat = max(
        state_people_per_seat) - min(state_people_per_seat)
    geo_mean_people_per_seat = calc_geo_mean(state_people_per_seat)

    mean_line.set_xdata([0, 1.0])
    mean_line.set_ydata([mean_people_per_seat])
    mean_txt.set_text(f"Mean: {mean_people_per_seat:,.2f}")
    std_dev_txt.set_text(f"Std. Dev.: {std_dev_people_per_seat:,.2f}")
    range_txt.set_text(f"Range: {range_people_per_seat:,.2f}")
    geo_mean_txt.set_text(f"Geo. Mean: {geo_mean_people_per_seat:,.2f}")

    seat_txt.set_text(f"Seat# {50 + frame}")
    state_txt.set_text(f"State: {max_state}")

    for bar, people_per_seat in zip(plt_1_bars, state_people_per_seat):
        bar.set_height(people_per_seat)

    # End Plot 1

    # Plot 2
    for bar, reps in zip(plt_2_bars, list(state_reps_name.values())):
        bar.set_height(reps)

    # End plot 2

    # Plot 3
    for bar, pritority_num in zip(plt_3_bars, state_priority_nums):
        bar.set_color("g")
        if pritority_num == state_priority_name[max_state]:
            bar.set_height(pritority_num)
            bar.set_color("r")
        else:
            bar.set_height(pritority_num)

    # End plot 3

    print("-" * 60)
    print(f"Seat# {frame}")
    print(f"Highest priority num: {max_state}")
    print("-" * 30)
    print(f"Priority nums: {state_priority_name}")
    print("-" * 30)
    print(f"State reps: {state_reps_name}")
    print("-" * 30)
    print(f"People per seat: {state_people_per_seat_name}")
    print("-" * 60)


Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='/u/ilikeplanes86'), bitrate=1800)

# account for frame zero
frames: int = 386
anim: Animation = animation.FuncAnimation(
    fig, animate, repeat=False, blit=False, frames=frames, interval=10)


figManager: Type[FigureManagerQT] = plt.get_current_fig_manager()
figManager.window.showMaximized()
figManager.set_window_title("CGP Grey Electoral College speadsheet animated")

plt.show()
# anim.save("bar-chart-autorecord.mp4", writer=writer)
