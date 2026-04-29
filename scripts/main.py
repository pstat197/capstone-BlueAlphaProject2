import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np

from scripts.config.loader import load_config
from scripts.spend_simulation.spend_generation import generate_spend_with_details
from scripts.spend_simulation.correlation_analysis import analyze_spend_correlations, print_correlation_report
from scripts.impressions_simulation.impressions_generation import generate_impressions
from scripts.revenue_simulation.revenue_generation import generate_revenue

from scripts.synth_input_classes.input_configurations import InputConfigurations


def construct_csv(
    config: InputConfigurations,
    spend_matrix: np.ndarray,  # matrix: weeks x channels
    impressions_matrix: np.ndarray,  # matrix: weeks x channels
    revenue_matrix: np.ndarray,  # matrix: weeks x channels
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

    data: Dict[str, Any] = {
        "week": np.arange(1, num_weeks + 1, dtype=int),
        "revenue": revenue_matrix.sum(axis=1).astype(float),
        "total_impressions": impressions_matrix.sum(axis=1).astype(float),
        "total_spend": spend_matrix.sum(axis=1).astype(float),
    }
    for i in range(num_channels):
        data[channel_impression_cols[i]] = impressions_matrix[:, i].astype(float)
        data[channel_spend_cols[i]] = spend_matrix[:, i].astype(float)
        data[channel_revenue_cols[i]] = revenue_matrix[:, i].astype(float)

    columns = ["week", "revenue"]
    for i in range(num_channels):
        columns += [
            channel_impression_cols[i],
            channel_spend_cols[i],
            channel_revenue_cols[i],
        ]
    columns += ["total_impressions", "total_spend"]
    df = pd.DataFrame(data)[columns]
    return df


def run_simulation(
    config: InputConfigurations,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Run spend -> impressions -> revenue and return (DataFrame, correlation_results).

    correlation_results includes both operational (post-toggle-mask) and generative
    (pre-toggle-mask) spend correlation analyses.
    """
    spend_pre_mask, spend_matrix = generate_spend_with_details(config)

    corr_results: Optional[Dict[str, Any]] = None
    if spend_matrix.size > 0 and spend_matrix.shape[1] >= 1:
        operational = analyze_spend_correlations(config, spend_matrix)
        generative = analyze_spend_correlations(config, spend_pre_mask)
        corr_results = {
            **operational,
            "operational_corr": operational,
            "generative_corr": generative,
            "correlation_basis_default": "operational",
        }

    impressions_matrix = generate_impressions(config, spend_matrix)
    revenue_by_channel = generate_revenue(config, impressions_matrix)
    df = construct_csv(config, spend_matrix, impressions_matrix, revenue_by_channel)
    return df, corr_results


def main(yaml_path):
    # Load config through the canonical loader path so CLI behavior matches UI/tests:
    # deep-merge defaults, auto-mode expansion, and validation all apply.
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = load_config(str(path))

    df, corr_results = run_simulation(config)

    if corr_results is not None:
        print_correlation_report(corr_results)

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
        parser.error(
            "Provide a YAML config path, e.g. python -m scripts.main example.yaml "
            "or python -m scripts.main -c path/to/config.yaml"
        )

    print(f"Running with config: {yaml_path}")
    print("--------------------------------")

    main(yaml_path)

    print("--------------------------------")
    print("Completed")