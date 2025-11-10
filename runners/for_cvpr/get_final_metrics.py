import json

from metrics.metrics_utils import aggregate_metrics
from runners.for_cvpr.settings import set_host_in_settings, Settings

if __name__ == "__main__":
    settings = Settings()
    settings = set_host_in_settings(settings)
    for use_gt_rot in [True, False]:
        metrics_dir = settings.paths.out_root / "metrics" / f"gt_poses-{use_gt_rot}"
        print(f"Getting metrics for {metrics_dir}")
        aggregate_metrics(metrics_dir)
