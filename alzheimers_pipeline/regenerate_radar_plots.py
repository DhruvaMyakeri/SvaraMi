import pandas as pd
from pathlib import Path
from radar_plots import make_all_radar_plots

OUTPUT_DIR = Path("outputs")

def main():
    deviation_csv = OUTPUT_DIR / "deviation_scores.csv"

    if not deviation_csv.exists():
        raise FileNotFoundError(
            f"Could not find {deviation_csv}"
        )

    deviation_df = pd.read_csv(deviation_csv)

    z_cols = [c for c in deviation_df.columns if c.startswith("z_")]
    feature_columns = [c[2:] for c in z_cols]

    if not feature_columns:
        raise ValueError(
            "No z_ columns found in deviation_scores.csv"
        )

    make_all_radar_plots(
        deviation_df=deviation_df,
        feature_columns=feature_columns,
        output_dir=OUTPUT_DIR,
    )

    print("Radar plots regenerated successfully.")

if __name__ == "__main__":
    main()