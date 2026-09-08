"""Read campaign progress without treating pending or failed blocks as results."""
import argparse
import json
from pathlib import Path

import psutil

from .campaign import read, expected_variants


def status(root):
    root = Path(root)
    spec = read(root/"campaign.json")
    manager = read(root/"state.json") if (root/"state.json").exists() else {"status": "initializing"}
    blocks, searches_done, searches_total = [], 0, 0
    for i in range(len(spec["block_seeds"])):
        folder = root/f"block_{i:02d}"
        cfg = read(folder/"config.json")
        s = read(folder/"state.json")
        done = len(list((folder/"runs").glob("*.json"))) if (folder/"runs").exists() else 0
        total = sum(t["split"] == "test" for t in read(folder/"tasks.json"))*len(cfg["search_seeds"])*len(expected_variants(cfg))
        searches_done += done
        searches_total += total
        entry = {"block": i, "status": s["status"], "saved_searches": done, "total_searches": total,
                 "updated_utc": s.get("updated_utc"), "worker_alive": psutil.pid_exists(s.get("pid", -1))
                 if s["status"] not in ("pending", "completed") else False}
        for key in ("task", "method", "model_seed", "tasks_completed", "tasks_total", "models_completed", "models_total", "verified_searches"):
            if key in s:
                entry[key] = s[key]
        if s["status"] == "training" and (folder/"training_progress.json").exists():
            progress = read(folder/"training_progress.json")
            if progress["loss"] == s.get("method") and progress["model_seed"] == s.get("model_seed"):
                entry["epoch"] = progress["epoch"]
                entry["epochs"] = progress["epochs"]
        blocks.append(entry)
    return {"campaign_status": manager["status"], "manager_alive": psutil.pid_exists(manager.get("pid", -1)),
            "saved_searches": searches_done, "total_searches": searches_total,
            "verified_blocks": sum((root/f"block_{i:02d}/COMPLETED.json").exists() for i in range(len(blocks))),
            "fully_completed": (root/"COMPLETED.json").exists(), "blocks": blocks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    print(json.dumps(status(parser.parse_args().out), indent=2, ensure_ascii=False))
