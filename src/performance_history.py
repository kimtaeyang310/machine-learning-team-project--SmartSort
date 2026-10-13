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


def draw_evaluation_history(history, path):
    """한 좌표계에 과일별 Macro F1 네 계열을 표시합니다."""
    series = [("overall", "전체", "#2563eb"), ("banana", "바나나", "#b77900"),
              ("mandarin", "귤", "#059669"), ("tomato", "토마토", "#dc2626")]
    frame = history[history["scope"].isin([s[0] for s in series])].copy()
    frame["time"] = pd.to_datetime(frame["evaluated_at"], utc=True)
    frame = frame.sort_values("time", kind="stable")
    times = sorted(frame["time"].unique())
    positions = {time: i for i, time in enumerate(times)}
    left, right, top, bottom = 80, 1135, 170, 535

    def x_for(time):
        return (left + right) / 2 if len(times) == 1 else left + (right - left) * positions[time] / max(1, len(times) - 1)

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="660" viewBox="0 0 1200 660" role="img" aria-label="전체, 바나나, 귤, 토마토의 평가 Macro F1 추이">',
           '<rect width="1200" height="660" rx="18" fill="#f8fafc"/>',
           '<g font-family="Segoe UI, Malgun Gothic, sans-serif" fill="#172033">',
           '<text x="40" y="44" font-size="26" font-weight="bold">숙도 분류 성능 추이</text>',
           '<text x="40" y="73" font-size="14" fill="#475569">Macro F1 · 높을수록 좋음 · 평가 실행 순서 / 한국 시간</text>']
    for i, (scope, name, color) in enumerate(series):
        data = frame[frame["scope"].eq(scope)]
        value = f'{float(data.iloc[-1]["macro_f1"]):.4f}' if not data.empty else "기록 없음"
        x = 45 + i * 288
        svg += [f'<line x1="{x}" y1="115" x2="{x + 27}" y2="115" stroke="{color}" stroke-width="4"/>',
                f'<text x="{x + 38}" y="121" font-size="17">{name}  <tspan fill="{color}" font-weight="bold">{value}</tspan></text>']
    for tick in range(11):
        value = tick / 10
        y = bottom - value * (bottom - top)
        svg += [f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#e2e8f0"/>',
                f'<text x="62" y="{y + 5}" text-anchor="end" font-size="13" fill="#64748b">{value:.1f}</text>']
    step = max(1, (len(times) + 7) // 8)
    for i, time in enumerate(times):
        if i % step and i != len(times) - 1:
            continue
        stamp = pd.Timestamp(time).tz_convert("Asia/Seoul")
        x = x_for(time)
        svg.append(f'<text x="{x}" y="563" text-anchor="middle" font-size="12" fill="#475569">{stamp:%m/%d}<tspan x="{x}" dy="18">{stamp:%H:%M}</tspan></text>')
    for scope, name, color in series:
        data = frame[frame["scope"].eq(scope)]
        svg.append(f'<g data-series="{scope}" stroke="{color}">')
        previous = None
        for row in data.itertuples():
            x, y = x_for(row.time), bottom - float(row.macro_f1) * (bottom - top)
            changed = previous is not None and previous[2] != row.dataset_id
            if previous is not None:
                dash = ' stroke-dasharray="6 5"' if changed else ''
                svg.append(f'<line x1="{previous[0]}" y1="{previous[1]}" x2="{x}" y2="{y}" stroke-width="2.5"{dash}/>')
            tooltip = escape(f"{name} | {row.evaluated_at} | Macro F1={row.macro_f1:.4f} | Accuracy={row.accuracy:.4f} | dataset={row.dataset_id}" + (" | 평가 데이터 변경" if changed else ""))
            svg.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}" stroke="#fff" stroke-width="1.5"><title>{tooltip}</title></circle>')
            previous = (x, y, row.dataset_id)
        svg.append('</g>')
    svg += ['<text x="40" y="615" font-size="13" fill="#475569">점 위에 마우스를 올리면 정확도·실행 시각·데이터 구분값을 확인할 수 있습니다.</text>',
            '<text x="40" y="638" font-size="13" fill="#475569">점선: 평가 데이터가 바뀐 구간입니다. 해당 구간의 점수 차이를 모델 개선만으로 해석하지 마세요.</text>',
            '</g></svg>']
    Path(path).write_text("\n".join(svg), encoding="utf-8")


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
    draw_evaluation_history(history, root / "evaluation_history.svg")
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
