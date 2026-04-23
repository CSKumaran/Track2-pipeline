"""Diagnostics writer for pipeline v2.2 — writes detailed intermediate outputs."""

import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DiagnosticsWriter:
    """Writes diagnostic JSON/CSV files to a diagnostics/ subfolder."""

    def __init__(self, output_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.diag_dir = os.path.join(output_dir, "diagnostics")
        if enabled:
            os.makedirs(self.diag_dir, exist_ok=True)
            logger.info("Diagnostics enabled: %s", self.diag_dir)

    def write_json(self, filename: str, data):
        """Write data as pretty-printed JSON."""
        if not self.enabled:
            return
        path = os.path.join(self.diag_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=self._json_default)
        logger.info("Diagnostic: %s", path)

    def write_csv(self, filename: str, df: pd.DataFrame):
        """Write DataFrame as CSV."""
        if not self.enabled:
            return
        path = os.path.join(self.diag_dir, filename)
        df.to_csv(path, index=False)
        logger.info("Diagnostic: %s (%d rows)", path, len(df))

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, set):
            return sorted(list(obj))
        return str(obj)
