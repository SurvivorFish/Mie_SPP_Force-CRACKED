import pandas as pd


def parse_file(filename: str):
    """
    parses comsol computation .csv files for different radiuses
    returns list of dfs each containing data for specific radius value
    """
    raw_data = pd.read_csv(filename, header=4)

    grouped = raw_data.groupby("% r_part (m)")
    out = [group for _, group in grouped]

    return out
