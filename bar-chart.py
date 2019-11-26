# TODO
# 1. Add type signatures

from typing import Type, Dict, List
from matplotlib.axes._subplots import Axes
from matplotlib.figure import Figure
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.container import BarContainer
from matplotlib.text import Text
from matplotlib.animation import Animation
from matplotlib.lines import Line2D
import math
import csv
import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker

with open("state-populations.csv") as inp:
    reader = csv.reader(inp)
    state_pops_name = {rows[0]: int(rows[1]) for rows in reader}
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


state_people_per_seat: List[float] = []
state_people_per_seat = calc_state_people_per_seat(state_pops, state_reps)

mean_people_per_seat: float = np.mean(state_people_per_seat)
std_dev_people_per_seat: float = np.std(state_people_per_seat)
state_priority_nums: List[float] = calc_priority_nums(
    state_names, state_reps_name, state_pops_name)
range_people_per_seat: float = 0

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
    0.35, 0.85, "State: ", transform=plt_1.transAxes)
mean_txt: Type[Text] = plt_1.text(
    0.45, 0.75, f"Mean: {mean_people_per_seat:,.2f}", transform=plt_1.transAxes)
std_dev_txt: Type[Text] = plt_1.text(
    0.55, 0.85, f"Std. Dev. {std_dev_people_per_seat}", transform=plt_1.transAxes)
range_txt: Type[Text] = plt_1.text(
    0.70, 0.75, f"Range: {range_people_per_seat}", transform=plt_1.transAxes)
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
plt_2.set_title("Number of representatives in each state")
plt_2_bars: Type[BarContainer] = plt_2.bar(
    y_pos, state_reps, align="center", alpha=0.5, color="r")
plt_2.set_xticks(x_pos)
plt_2.set_xticklabels(state_names, rotation=77)

plt_2.set_ylabel("Representatives")
plt_2.set_ylim(60)
# Y-axis gets flpped for some reason
plt_2.invert_yaxis()


plt_3_bars: Type[BarContainer] = plt_3.bar(y_pos, state_priority_nums,
                                           align="center", alpha=0.5, color="g")
plt_3.set_xticks(x_pos)
plt_3.set_xticklabels(state_names, rotation=77)
plt_3.set_title("Priority values for each state")

plt_3.set_ylabel("Priority value")
plt_3.set_yscale("log")
plt_3.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt_3.text(0.3, 0.9, "Highlighted, is the state with the highest priority value",
           transform=plt_3.transAxes)


plt.subplots_adjust(top=0.964,
                    bottom=0.138,
                    left=0.064,
                    right=0.986,
                    hspace=0.456,
                    wspace=0.072)


def animate(frame: int) -> None:
    if frame < 2:
        return

    print("-" * 45)
    print(f"Seat# {frame}")

    # Plot 1
    state_priority_nums = calc_priority_nums(
        state_names, state_reps_name, state_pops_name)
    state_priority_name = dict(zip(state_names, state_priority_nums))
    print(f"Priority nums: {state_priority_name}")

    max_state: str = max(state_priority_name.items(),
                         key=operator.itemgetter(1))[0]
    print(f"Highest priority num: {max_state}")

    state_reps_name[max_state] = state_reps_name[max_state] + 1
    print(f"State reps: {state_reps_name}")

    state_people_per_seat = []
    state_people_per_seat = calc_state_people_per_seat(
        state_pops, list(state_reps_name.values()))

    mean_people_per_seat = np.mean(state_people_per_seat)
    std_dev_people_per_seat = np.std(state_people_per_seat)
    range_people_per_seat = max(
        state_people_per_seat) - min(state_people_per_seat)

    mean_line.set_xdata([0, 1.0])
    mean_line.set_ydata([mean_people_per_seat])
    mean_txt.set_text(f"Mean: {mean_people_per_seat:,.2f}")
    std_dev_txt.set_text(f"Std. Dev.: {std_dev_people_per_seat:,.2f}")
    range_txt.set_text(f"Range: {range_people_per_seat:,.2f}")

    print(f"People per seat: {dict(zip(state_names, state_people_per_seat))}")

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

    print("-" * 45)


# account for frame zero
frames: int = 386
anim: Animation = animation.FuncAnimation(
    fig, animate, repeat=False, blit=False, frames=frames, interval=190)

figManager: Type[FigureManagerBase] = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
