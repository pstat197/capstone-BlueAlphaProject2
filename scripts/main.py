import argparse
import os
from datetime import datetime

import pandas as pd
import numpy as np

from scripts.spend_simulation.spend_generation import generate_spend
from scripts.impressions_simulation.impressions_generation import generate_impressions
from scripts.revenue_simulation.revenue_generation import generate_revenue
from scripts.config.loader import load_config
from scripts.synth_input_classes.input_configurations import InputConfigurations


def construct_csv(
    config: InputConfigurations,
    spend_matrix: "np.ndarray",  # matrix: weeks x channels
    impressions_matrix: "np.ndarray",  # matrix: weeks x channels
    revenue_matrix: "np.ndarray",  # matrix: weeks x channels
) -> pd.DataFrame:
    """
    Construct a DataFrame: each row corresponds to a week.
    Columns:
      - 'week': week number (starting from 1)
      - 'revenue': total revenue for the week (sum across channels)
      - For each channel: f"{channel}_impressions", f"{channel}_spend", f"{channel}_revenue"
      - 'total_impressions', 'total_spend'
    """
    # Get channel names and week count
    channel_names = [ch.get_channel_name() for ch in config.get_channel_list()]
    num_weeks = spend_matrix.shape[0]
    num_channels = spend_matrix.shape[1]

    # Prepare column names for each channel
    channel_impression_cols = [f"{name}_impressions" for name in channel_names]
    channel_spend_cols = [f"{name}_spend" for name in channel_names]
    channel_revenue_cols = [f"{name}_revenue" for name in channel_names]

    # Build rows
    data = []
    for week in range(num_weeks):
        row: dict = {"week": week + 1}
        total_impressions = 0
        total_spend = 0
        for ch in range(num_channels):
            imp = impressions_matrix[week, ch]
            spd = spend_matrix[week, ch]
            row[channel_impression_cols[ch]] = imp
            row[channel_spend_cols[ch]] = spd
            row[channel_revenue_cols[ch]] = float(revenue_matrix[week, ch])
            total_impressions += imp
            total_spend += spd
        row["revenue"] = float(revenue_matrix[week].sum())
        row["total_impressions"] = total_impressions
        row["total_spend"] = total_spend
        data.append(row)

    # One block per channel: impressions, spend, revenue (easier to read in CSV / print(df.head()))
    columns = ["week", "revenue"]
    for i in range(num_channels):
        columns += [
            channel_impression_cols[i],
            channel_spend_cols[i],
            channel_revenue_cols[i],
        ]
    columns += ["total_impressions", "total_spend"]
    df = pd.DataFrame(data, columns=columns)
    return df


def run_simulation(config: InputConfigurations) -> pd.DataFrame:
    """Run spend → impressions → revenue and return the weekly DataFrame."""
    spend_matrix = generate_spend(config)
    impressions_matrix = generate_impressions(config, spend_matrix)
    revenue_by_channel = generate_revenue(config, impressions_matrix)
    return construct_csv(config, spend_matrix, impressions_matrix, revenue_by_channel)


def main(yaml_path):

    # load config (merges with config/default.yaml; supports number_of_channels)
    config = load_config(yaml_path)

    df = run_simulation(config)

    print(df.head())
    print("Columns:", ", ".join(map(str, df.columns)))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs("output", exist_ok=True)
    output_path = f"output/{config.get_run_identifier()}_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run with a YAML config (e.g. example.yaml)")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "-c", "--config-file",
        dest="config_file",
        default=None,
        help="Path to YAML config file (alternative to positional)",
    )
    args = parser.parse_args()
    yaml_path = args.config or args.config_file

    if not yaml_path:
        parser.error("Provide a YAML config path, e.g. python main.py example.yaml or python main.py -c path/to/config.yaml")

    print(f"Running with config: {yaml_path}")
    print("--------------------------------")

    main(yaml_path)

    print("--------------------------------")
    print("Completed")