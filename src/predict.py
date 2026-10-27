"""Joint YOLO prediction entry points and CSV/manifest export."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.decision import DEFAULT_CONFIDENCE_THRESHOLD, REVIEW_REQUIRED_CLASSES

PREDICT_DIR = PROJECT_ROOT / "data" / "predict_list"
MODEL_PATH = PROJECT_ROOT / "models" / "joint_nine" / "model.pt"


def predict(image_path):
    from src.joint import JointClassifier
    result = JointClassifier().predict(image_path)
    print("모르는 과일" if result["status"] == "unknown" else f"{result['fruit']} / {result['ripeness']}")
    return result["ripeness"] if result["status"] == "ok" else "unknown"


def predict_folder(folder_path=PREDICT_DIR):
    from src.joint import predict_folder as joint_predict_folder
    return joint_predict_folder(folder_path)


def save_prediction_artifacts(results, output_csv, *, model_path=None,
                              source_folder=PREDICT_DIR):
    """예측·에이전트 결과 CSV와 실행 환경 manifest를 저장합니다."""
    output_csv = Path(output_csv)
    if not output_csv.is_absolute():
        output_csv = PROJECT_ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "image_path", "status", "fruit", "model", "ripeness",
        "probabilities_json", "agent_decision_json", "error",
        "preprocessing_ms_json", "feature_ms", "prediction_ms", "total_ms",
        "fruit_identity_json", "fruit_identity_ms",
        "nine_probabilities_json", "candidate_class", "rejection_score", "rejection_threshold",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "image_path": result.get("image_path", ""),
                "status": result.get("status", ""),
                "fruit": result.get("fruit", ""),
                "model": result.get("model", ""),
                "ripeness": result.get("ripeness", ""),
                "probabilities_json": json.dumps(
                    result.get("probabilities"), ensure_ascii=False, sort_keys=True
                ),
                "agent_decision_json": json.dumps(
                    result.get("agent_decision"), ensure_ascii=False, sort_keys=True
                ),
                "error": result.get("error", ""),
                "preprocessing_ms_json": json.dumps(
                    result.get("preprocessing_ms"), ensure_ascii=False, sort_keys=True
                ),
                "feature_ms": result.get("feature_ms", ""),
                "prediction_ms": result.get("prediction_ms", ""),
                "total_ms": result.get("total_ms", ""),
                "fruit_identity_json": json.dumps(result.get("fruit_identity"), ensure_ascii=False, sort_keys=True),
                "fruit_identity_ms": result.get("fruit_identity_ms", ""),
                "nine_probabilities_json": json.dumps(result.get("nine_probabilities"), ensure_ascii=False, sort_keys=True),
                "candidate_class": result.get("candidate_class", ""),
                "rejection_score": result.get("score", ""),
                "rejection_threshold": result.get("threshold", ""),
            })

    if model_path is None:
        model_path = (PROJECT_ROOT / "models/joint_nine/model.pt"
                      if any(str(r.get("model", "")).endswith("_cls_joint_nine") for r in results) else MODEL_PATH)
    model_path = Path(model_path)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_folder": str(Path(source_folder).resolve()),
        "model_bundle_path": str(model_path.resolve()),
        "model_bundle_sha256": (
            hashlib.sha256(model_path.read_bytes()).hexdigest()
            if model_path.is_file() else None
        ),
        "models_used": sorted({
            f"{row.get('fruit')}: {row.get('model')}"
            for row in results if row.get("status") == "ok"
        }),
        "agent_policy": {
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "human_review_classes": sorted(
                [fruit, ripeness] for fruit, ripeness in REVIEW_REQUIRED_CLASSES
            ),
        },
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "result_count": len(results),
        "fruit_identity_models": sorted({row["fruit_identity"]["model_sha256"] for row in results
                                          if row.get("fruit_identity")}),
        "unknown_count": sum(row.get("status") == "unknown" for row in results),
        "failure_count": sum(row.get("status") == "failed" for row in results),
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    if model_path == PROJECT_ROOT / "models/joint_nine/model.pt":
        manifest["joint_bundle"] = json.loads((model_path.parent / "bundle.json").read_text(encoding="utf-8"))
    if any(str(r.get("model", "")).endswith("_cls_joint_nine") for r in results):
        from src.joint import timing_summary
        manifest["timing"] = timing_summary(results)
        manifest["timing_protocol"] = "warmup excluded; image read, preprocessing, YOLO and rejection included; agent, printing and saving excluded"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_csv, manifest_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="통합 YOLO 폴더 예측 및 시간 측정")
    parser.add_argument("folder", nargs="?", default=str(PREDICT_DIR))
    parser.add_argument("--results-csv", type=Path,
                        default=PROJECT_ROOT / "outputs" / "predictions" /
                        (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".csv"))
    args = parser.parse_args()
    try:
        results, _ = predict_folder(args.folder)
        csv_path, manifest_path = save_prediction_artifacts(results, args.results_csv, source_folder=args.folder)
        print(f"결과: {csv_path}\n실행 정보: {manifest_path}")
    except (ValueError, OSError, RuntimeError) as error:
        print(f"작업 중단: {error}")
        raise SystemExit(1)
