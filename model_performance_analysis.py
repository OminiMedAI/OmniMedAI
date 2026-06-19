"""Compatibility CLI for evaluating saved prediction tables."""

import argparse
import json


def main():
    import pandas as pd

    from onem_eval import bootstrap_auc_ci, calibration_table, decision_curve

    parser = argparse.ArgumentParser(description="Evaluate binary predictions")
    parser.add_argument("predictions_csv")
    parser.add_argument("--truth", default="y_true")
    parser.add_argument("--score", default="y_score")
    parser.add_argument("--output-prefix", default="model_evaluation")
    args = parser.parse_args()
    table = pd.read_csv(args.predictions_csv)
    auc = bootstrap_auc_ci(table[args.truth], table[args.score])
    with open(f"{args.output_prefix}_auc.json", "w", encoding="utf-8") as handle:
        json.dump(auc, handle, indent=2)
    calibration_table(table[args.truth], table[args.score]).to_csv(
        f"{args.output_prefix}_calibration.csv", index=False
    )
    decision_curve(table[args.truth], table[args.score]).to_csv(
        f"{args.output_prefix}_dca.csv", index=False
    )


if __name__ == "__main__":
    main()
