"""실행별 CSV를 모아 이력과 의존성 없는 SVG 비교 그래프를 저장합니다."""

from datetime import datetime, timezone, timedelta
from hashlib import sha256
from html import escape
from pathlib import Path

import pandas as pd


def run_time():
    return datetime.now(timezone(timedelta(hours=9)))


def sample_id(frame):
    """행 순서와 무관한 사진 목록·정답·그룹 식별자 (이미지 내용 해시는 아님)."""
    columns = ["image_path", "fruit", "group_id", "ripeness_label"]
    canonical = frame[columns].astype(str).sort_values(columns).to_csv(index=False)
    return sha256(canonical.encode("utf-8")).hexdigest()[:12]


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def draw_history(history, path, series_column, metrics, title):
    """같은 대상·같은 데이터 목록만 선으로 연결합니다. 점수 축은 0~1입니다."""
    history = history.copy()
    history["run_id"] = history["evaluated_at"]
    runs = history[["run_id", "evaluated_at"]].drop_duplicates().sort_values("run_id")
    ids = runs["run_id"].tolist()
    width = max(1000, 180 + 110 * len(ids))
    panels = list(history.groupby([series_column, "dataset_id"], sort=True))
    height = 95 + 340 * len(panels)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#f8fafc"/>',
           '<g font-family="Segoe UI, Arial, sans-serif" fill="#172033">',
           f'<text x="35" y="35" font-size="23" font-weight="bold">{escape(title)}</text>',
           '<text x="35" y="62" font-size="13">KST (UTC+09:00) | Separate panels = different sample lists / labels. Hover points for details.</text>']
    colors = ["#2563eb", "#d97706"]
    for panel, ((scope, dataset), frame) in enumerate(panels):
        top = 115 + panel * 340
        left, right, bottom = 75, width - 45, top + 195
        svg.append(f'<text x="35" y="{top - 20}" font-size="16" font-weight="bold">{escape(str(scope))} | dataset {escape(str(dataset))}</text>')
        for tick in range(6):
            value = tick / 5
            y = bottom - value * 180
            svg += [f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#dce3ed"/>',
                    f'<text x="35" y="{y + 4}" font-size="12">{value:.1f}</text>']
        def x_for(run):
            return left + (right - left) * (ids.index(run) + 0.5) / len(ids)
        for run, timestamp in runs.itertuples(index=False, name=None):
            x = x_for(run)
            label = str(timestamp).replace("T", " ").split("+")[0]
            svg.append(f'<text transform="translate({x},{bottom + 18}) rotate(25)" font-size="11">{escape(label)}</text>')
        frame = frame.sort_values("run_id")
        for metric, color in zip(metrics, colors):
            points = [(x_for(row.run_id), bottom - float(getattr(row, metric)) * 180, row)
                      for row in frame.itertuples()]
            coordinates = " ".join(f"{x},{y}" for x, y, _ in points)
            svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y, row in points:
                value = float(getattr(row, metric))
                tooltip = escape(f"{row.evaluated_at} | {metric}={value:.4f}")
                svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"><title>{tooltip}</title></circle>')
            legend_x = 80 + metrics.index(metric) * 250
            svg.append(f'<text x="{legend_x}" y="{top + 285}" fill="{color}" font-size="13">● {escape(metric)}</text>')
    svg.append('</g></svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


EVALUATION_COLUMNS = ["evaluated_at", "scope", "dataset_id", "accuracy", "macro_f1"]


def rebuild_evaluation_history(output_root, current=None):
    """완료된 평가만 집계하며, 기존 형식도 원래 실행 시각으로 가져옵니다."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    tables = []
    history_path = root / "evaluation_history.csv"
    if history_path.exists():
        tables.append(pd.read_csv(history_path, keep_default_na=False)[EVALUATION_COLUMNS])
    for metrics_path in sorted(root.glob("*/metrics.csv")):
        run = metrics_path.parent
        table = pd.read_csv(metrics_path, keep_default_na=False)
        if "evaluated_at" not in table:
            try:
                timestamp = datetime.strptime(run.name, "%Y%m%d_%H%M%S_%f")
            except ValueError:
                continue
            table["evaluated_at"] = timestamp.isoformat() + "+09:00"
        table["run_id"] = run.name
        if "model_name" not in table:
            table["model_name"] = "unknown (legacy)"
        if "dataset_id" not in table:
            samples_path = run / "test_samples.csv"
            if samples_path.exists():
                samples = pd.read_csv(samples_path, keep_default_na=False)
                table["dataset_id"] = [sample_id(samples if scope == "overall" else samples[samples.fruit == scope])
                                       for scope in table.scope]
            else:
                table["dataset_id"] = "unknown-" + run.name
        tables.append(table)
    if not tables:
        if current is None:
            return None
        tables.append(current[EVALUATION_COLUMNS])
    history = pd.concat(tables, ignore_index=True)[EVALUATION_COLUMNS]
    if current is not None:
        history = pd.concat([history, current[EVALUATION_COLUMNS]], ignore_index=True)
    history = history.drop_duplicates(["evaluated_at", "scope", "dataset_id"], keep="last")
    history = history.sort_values(["evaluated_at", "scope"]).reset_index(drop=True)
    save_csv(history, history_path)
    draw_history(history, root / "evaluation_history.svg", "scope",
                 ["accuracy", "macro_f1"], "Test evaluation history")
    return history


def record_training(comparison, samples, output_root):
    root = Path(output_root)
    now = run_time()
    run = root / "runs" / now.strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    table = comparison.copy()
    table["run_id"] = run.name
    table["evaluated_at"] = now.isoformat(timespec="seconds")
    table["dataset_id"] = sample_id(pd.DataFrame(samples))
    table["n_images"] = len(samples)
    table["selected"] = table["model"].eq(comparison.iloc[0]["model"])
    save_csv(table, run / "model_comparison.csv")
    files = sorted((root / "runs").glob("*/model_comparison.csv"))
    history = pd.concat([pd.read_csv(p, keep_default_na=False) for p in files], ignore_index=True)
    save_csv(history, root / "training_history.csv")
    draw_history(history, root / "training_history.svg", "model",
                 ["cv_macro_f1_mean"], "Training cross-validation history (not Test scores)")
    return run


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "outputs" / "evaluation"
    rebuild_evaluation_history(root)
    print(f"평가 이력 및 그래프 갱신: {root}")
