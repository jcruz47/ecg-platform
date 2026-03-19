import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.db.supabase_client import get_supabase
from app.services.image_full_pipeline import run_full_image_pipeline


MANIFEST_PATH = Path("data/ptbxl_paired/manifests/studies_master.csv")
OUT_CSV = Path("data/ptbxl_paired/manifests/eval_reconstruction_results.csv")

# Para empezar, evalúa pocas muestras
MAX_SAMPLES = 100


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_abs_diff(a, b):
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def main():
    sb = get_supabase()
    df = pd.read_csv(MANIFEST_PATH)

    # Puedes empezar por full + clean
    df = df[df["variant"] == "clean"].copy()
    df = df.head(MAX_SAMPLES)

    rows = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        image_path = row["image_path"]
        metrics_gt_path = row["metrics_path"]
        rpeaks_gt_path = row["rpeaks_path"]

        raw_bytes = Path(image_path).read_bytes()

        try:
            payload = run_full_image_pipeline(sb, "eval-paired", raw_bytes)
        except Exception as e:
            rows.append({
                "sample_id": sample_id,
                "image_label_gt": row["image_label"],
                "duration_sec_gt": row["duration_sec"],
                "fs_gt": row["fs_gt"],
                "error": str(e),
            })
            continue

        gt_metrics = load_json(metrics_gt_path)
        gt_rpeaks = load_json(rpeaks_gt_path)

        qc = payload.get("qc", {})
        img_cls = payload.get("image_classification", {}) or {}
        rec = payload.get("reconstruction", {}) or {}
        sig = payload.get("signal_payload", {}) or {}
        pred_metrics = sig.get("metrics") or {}
        pred_summary = sig.get("summary") or {}
        pred_findings = sig.get("findings") or []

        gt_hr = gt_metrics.get("heart_rate_bpm")
        pred_hr = pred_metrics.get("heart_rate_bpm")

        gt_rpeaks_count = len(gt_rpeaks.get("rpeaks_samples", []))
        pred_rpeaks_count = pred_summary.get("rpeaks_detected")

        rows.append({
            "sample_id": sample_id,
            "image_label_gt": row["image_label"],
            "duration_sec_gt": row["duration_sec"],
            "fs_gt": row["fs_gt"],

            "qc_label": qc.get("quality_label"),
            "qc_usable": qc.get("usable"),

            "selected_variant": payload.get("selected_preprocess_variant"),
            "reconstruction_quality_score": rec.get("reconstruction_quality_score"),

            "image_pred_label": img_cls.get("predicted_label"),
            "image_pred_conf": img_cls.get("confidence"),

            "gt_hr_bpm": gt_hr,
            "pred_hr_bpm": pred_hr,
            "abs_hr_error": safe_abs_diff(gt_hr, pred_hr),

            "gt_rpeaks_count": gt_rpeaks_count,
            "pred_rpeaks_count": pred_rpeaks_count,
            "abs_rpeaks_count_error": safe_abs_diff(gt_rpeaks_count, pred_rpeaks_count),

            "pred_duration_sec": pred_summary.get("duration_seconds"),
            "pred_fs": pred_summary.get("sampling_rate"),

            "signal_error": sig.get("error"),
            "pred_findings": json.dumps(pred_findings, ensure_ascii=False),
        })

        print(f"[OK] {sample_id}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print("\nResultados guardados en:", OUT_CSV.resolve())
    print("Total evaluadas:", len(out))

    if "abs_hr_error" in out.columns:
        valid_hr = out["abs_hr_error"].dropna()
        if len(valid_hr) > 0:
            print("MAE HR:", valid_hr.mean())

    if "abs_rpeaks_count_error" in out.columns:
        valid_r = out["abs_rpeaks_count_error"].dropna()
        if len(valid_r) > 0:
            print("MAE count R-peaks:", valid_r.mean())


if __name__ == "__main__":
    main()