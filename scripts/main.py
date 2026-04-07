import argparse
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

from scripts.spend_simulation.spend_generation import generate_spend
from scripts.spend_simulation.correlation_analysis import analyze_spend_correlations, print_correlation_report
from scripts.impressions_simulation.impressions_generation import generate_impressions
from scripts.revenue_simulation.revenue_generation import generate_revenue

from scripts.synth_input_classes.input_configurations import InputConfigurations


def construct_csv(
    config: InputConfigurations,
    spend_matrix: "np.ndarray",  # matrix: weeks x channels
    impressions_matrix: "np.ndarray",  # matrix: weeks x channels
    revenue_vector: "np.ndarray",  # vector: weeks
) -> pd.DataFrame:
    """
    Construct a DataFrame: each row corresponds to a week.
    Columns:
      - 'week': week number (starting from 1)
      - 'revenue': total revenue for the week
      - For each channel: f"{channel}_impressions" and f"{channel}_spend"
      - 'total_impressions', 'total_spend'
    """
    # Get channel names and week count
    channel_names = [ch.get_channel_name() for ch in config.get_channel_list()]
    num_weeks = spend_matrix.shape[0]
    num_channels = spend_matrix.shape[1]

    # Prepare column names for each channel
    channel_impression_cols = [f"{name}_impressions" for name in channel_names]
    channel_spend_cols = [f"{name}_spend" for name in channel_names]

    # Build rows
    data = []
    for week in range(num_weeks):
        row = {
            "week": week + 1,
            "revenue": revenue_vector[week],
        }
        total_impressions = 0
        total_spend = 0
        for ch in range(num_channels):
            imp = impressions_matrix[week, ch]
            spd = spend_matrix[week, ch]
            row[channel_impression_cols[ch]] = imp
            row[channel_spend_cols[ch]] = spd
            total_impressions += imp
            total_spend += spd
        row["total_impressions"] = total_impressions
        row["total_spend"] = total_spend
        data.append(row)

    columns = (
        ["week", "revenue"]
        + [col for pair in zip(channel_impression_cols, channel_spend_cols) for col in pair]
        + ["total_impressions", "total_spend"]
    )
    df = pd.DataFrame(data, columns=columns)
    return df

def main(yaml_path):

    # load config
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    config = InputConfigurations.from_yaml_dict(data)

    # generate spend
    # spend_matrix = generate_spend(config)

    # correlation analysis (runs after spend, before impressions)
    # if config.get_correlations():
    #     corr_results = analyze_spend_correlations(config, spend_matrix)
    #     print_correlation_report(corr_results)

    # generate impressions
    # impressions_matrix = generate_impressions(config, spend_matrix)

    # generate revenue
    # revenue_matrix = generate_revenue(config, impressions_matrix)

    # construct csv
    # df = construct_csv(config, spend_matrix, impressions_matrix, revenue_matrix)
    # df.to_csv(f"output/{config.get_run_identifier()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)
    # print(f"Saved to: output/{config.get_run_identifier()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

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