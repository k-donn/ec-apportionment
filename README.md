# ec-apportionment

A Matplotlib representation of CGP Grey's Electoral College spreadsheet. This animates
the number of representatives, people to representative ratio, and priority
number calculations.

See [Running](#running) for instructions on how to get started

## Running

-   Create conda env from requirements.txt (see [Install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))
-   From the root of the folder,

```
python3 ./source/bar-chart.py ./data/<STATE-POPULATIONS>.csv
```

-   If there are any errors, they are most likely backend related.
-   Adjust code to use whatever backend you have on your system (project is Qt5Agg)

## Meta

From this [video](https://www.youtube.com/watch?v=6JN4RI7nkes).

Update/Change the state population data by putting your own data into [state-populations.csv](https://github.com/k-donn/ec-apportionment/blob/master/data/state-populations.csv) or pass the new name of the file from the command line.

The CSV file should not have a header.
