"""WhisperX timestamp sanity checker [v2.1]."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def validate_timestamps(words_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate word timestamps from WhisperX.
    Adds 'timestamp_reliable' column.

    Checks:
      a. Word duration: 0.05s < duration < 2.0s
      b. Gap > 3s between consecutive words not flagged as silence
      c. Monotonically increasing timestamps
    """
    df = words_df.copy()
    df["timestamp_reliable"] = True

    if df.empty:
        return df

    # Ensure numeric
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")

    n_flagged = 0

    for i in range(len(df)):
        start = df.iloc[i]["start_time"]
        end = df.iloc[i]["end_time"]

        # Check a: duration
        if pd.notna(start) and pd.notna(end):
            duration = end - start
            if duration < 0.05 or duration > 2.0:
                df.at[df.index[i], "timestamp_reliable"] = False
                n_flagged += 1
                continue
        else:
            df.at[df.index[i], "timestamp_reliable"] = False
            n_flagged += 1
            continue

        # Check b: gap from previous word
        if i > 0:
            prev_end = df.iloc[i - 1]["end_time"]
            if pd.notna(prev_end) and pd.notna(start):
                gap = start - prev_end
                if gap > 3.0:
                    df.at[df.index[i], "timestamp_reliable"] = False
                    n_flagged += 1
                    continue

        # Check c: monotonicity
        if i > 0:
            prev_start = df.iloc[i - 1]["start_time"]
            if pd.notna(prev_start) and pd.notna(start):
                if start < prev_start:
                    df.at[df.index[i], "timestamp_reliable"] = False
                    n_flagged += 1
                    continue

    pct = (n_flagged / len(df)) * 100 if len(df) > 0 else 0
    logger.info("Timestamp sanity: %d/%d words flagged (%.1f%%)", n_flagged, len(df), pct)
    return df
