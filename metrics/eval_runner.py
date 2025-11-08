
from metrics.metrics_utils import aggregate_metrics
from runners.run_argus_cotracker import Settings

USE_GT_POSES = True # TODO put in settings
if __name__ == "__main__":
    settings = Settings()

    out_root = settings.out_root / "argus_cotracker_outputs" / settings.ds_name
    if settings.ds_name == "tapvid360-10k":
        out_root = out_root / f"gt_poses-{USE_GT_POSES}"
    metrics_dir = out_root / "metrics"

    print(f"\nAggregating metrics")
    aggregate_metrics(metrics_dir)
