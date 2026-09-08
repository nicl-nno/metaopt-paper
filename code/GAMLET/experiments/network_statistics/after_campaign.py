"""Run the unchanged statistical analysis once the entire campaign is audited."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def state(folder, status, **extra):
    data = {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), **extra}
    tmp = folder/"state.tmp"
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for attempt in range(100):
        try:
            os.replace(tmp, folder/"state.json")
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def run(campaign, out, monitor):
    monitor.mkdir(parents=True, exist_ok=False)
    while not (campaign/"COMPLETED.json").exists():
        if (campaign/"FAILED.json").exists():
            state(monitor, "campaign_failed", error=str(campaign/"FAILED.json"))
            return
        state(monitor, "waiting_for_all_five_audited_blocks", campaign=str(campaign.resolve()))
        time.sleep(30)
    if out.exists():
        state(monitor, "output_already_exists", output=str(out.resolve()))
        return
    state(monitor, "calculating_final_tests", output=str(out.resolve()))
    with (monitor/"analysis.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("analyze.py")),
                                 "--campaign", str(campaign), "--out", str(out)],
                                stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    state(monitor, "completed" if result.returncode == 0 and (out/"COMPLETED.json").exists() else "analysis_failed",
          exit_code=result.returncode, output=str(out.resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    args = parser.parse_args()
    run(args.campaign, args.out, args.monitor)
