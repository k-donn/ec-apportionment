"""
Show an animation of the Huntington–Hill apportionment method.

usage: python3.8 run.py [-h] -f FILE [-d]

required arguments:
  -f FILE, --file FILE  Path to CSV state population data

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           Show the plot instead of writing to file

"""

from source import bar_chart

from argparse import ArgumentParser


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        prog="python3.8 run.py",
        description="Show an animation of the Huntington–Hill apportionment method")
    parser.add_argument("-f", "--file", required=True,
                        help="Path to CSV state population data")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Show the plot instead of writing to file")

    args = parser.parse_args()

    bar_chart.main(args.file, args.debug)
