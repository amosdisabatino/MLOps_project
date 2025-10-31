import pandas as pd
import logging
import os
import json
from datetime import datetime
from src.config import CURRENT_PATH, LABELS, REF_STATS_FILE, OUTPUT_REPORT
import subprocess
import sys

os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_log():
    df = pd.read_csv(CURRENT_PATH, parse_dates=['Timestamp'])
    df['Label'] = df['Label'].map(LABELS)
    return df


def compute_current_stats(df):
    stats = {}
    stats['count'] = len(df)
    stats['mean_confidence'] = df['Confidence'].mean()
    stats['pred_dist'] = df['Label'].value_counts(normalize=True).to_dict()
    return stats


def load_reference_stats():
    if os.path.exists(REF_STATS_FILE):
        return json.load(open(REF_STATS_FILE))
    else:
        return None


def compare_stats(
    ref,
    curr,
    data_threshold=100,
    threshold_conf=0.7,
    threshold_dist_change=0.2,
):
    """
    This method is used to compare reference and current statistics to
    identify significant changes.
    Args:
        ref (dict): Reference statistics.
        curr (dict): Current statistics.
        data_threshold (int): Minimum number of records to consider.
        threshold_conf (float): Threshold for mean confidence alert.
        threshold_dist_change (float): Threshold for predicted distribution
        change alert.
    """
    alerts = []
    if curr['count'] >= data_threshold:
        if curr['mean_confidence'] < threshold_conf:
            alerts.append(
                f"Mean confidence {curr['mean_confidence']:.3f} "
                f"below threshold {threshold_conf}"
            )
        if ref:
            for label, frac in curr["pred_dist"].items():
                ref_frac = ref["pred_dist"].get(str(label), 0)
                # This check is made only if there is a reference stats.
                if ref_frac and abs(frac - ref_frac) > threshold_dist_change:
                    alerts.append(
                        f"Predicted class {label} fraction changed from "
                        f"{ref_frac:.3f} to {frac:.3f}"
                    )
                else:
                    logger.info(
                        f"No significant change for class {label}: "
                        f"{ref_frac:.3f} to {frac:.3f}"
                    )
    else:
        logger.info(
            f"Not enough data for monitoring: {curr['count']} records "
            f"found, {data_threshold} required."
        )
    return alerts


def save_current_as_reference(curr):
    with open(REF_STATS_FILE, "w") as f:
        json.dump(curr, f)


def start_training():
    train_script = os.path.join("src", "train.py")
    if os.path.exists(train_script):
        try:
            logger.info(
                f"Alerts detected — launching training script: {train_script}"
            )
            subprocess.run(
                [sys.executable, train_script],
                check=True,
            )
            logger.info('Training script finished successfully.')
        except subprocess.CalledProcessError as e:
            logger.exception(
                f"Training script exited with non-zero status: {e}"
            )
        except Exception as e:
            logger.exception(f"Failed to run training script: {e}")
    else:
        logger.error(f"Training script not found at {train_script}")


def main():

    logger.info("Starting Monitoring Process...")

    df = load_log()
    curr = compute_current_stats(df)
    ref = load_reference_stats()

    report = {
        "timestamp": datetime.now().isoformat(),
        "current_stats": curr,
        "reference_stats": ref,
    }

    alerts = compare_stats(ref, curr)
    report['alerts'] = alerts

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f'Report: {json.dumps(report, indent=2)}')

    if alerts and ref:
        # If `alerts` and `ref` exist, start retraining.
        start_training()

    # if ref not exists, save the first one as reference
    if ref is None:
        save_current_as_reference(curr)
        logger.info("Reference stats initialized.")


if __name__ == "__main__":
    main()
