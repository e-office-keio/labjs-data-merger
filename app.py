import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="lab.js データ結合ツール", layout="wide"
)

st.title("🧪 lab.js 実験データ結合アプリ")
st.write(
    "複数のCSVファイルをアップロードすると、1被験者1行（ワイド形式）にまとめてダウンロードできます。"
)

# ファイルアップローダー（複数ファイル・ドラッグ＆ドロップ対応）
uploaded_files = st.file_uploader(
    "CSVファイルを選択、またはここにドラッグ＆ドロップしてください",
    type=["csv"],
    accept_multiple_files=True,
)


def get_first_valid(df, col_name):
    """指定したカラムの最初の非空値を取り出すヘルパー関数"""
    if col_name in df.columns:
        valid_vals = df[col_name].dropna().values
        for val in valid_vals:
            if pd.notna(val) and str(val).strip() != "":
                return val
    return np.nan


def process_single_file(file):
    """1つのファイル（1被験者分）を1行のデータに変換する関数"""
    df = pd.read_csv(file)

    # 1. 被験者基本情報
    p_id = get_first_valid(df, "participantID")
    if pd.isna(p_id):
        p_id = file.name
    else:
        # IDが数値の場合整数形式の文字列に整える
        try:
            p_id = str(int(float(p_id)))
        except (ValueError, TypeError):
            p_id = str(p_id)

    age = get_first_valid(df, "age")
    sex = get_first_valid(df, "sex")
    condition = get_first_valid(df, "condition")

    # 2. クイズ・集計指標
    quiz_correct = get_first_valid(df, "quiz_correct_count")
    quiz_total = get_first_valid(df, "quiz_total_count")

    # クイズ正答率・平均反応時間計算 (Showquiz 画面の試行データから)
    quiz_df = df[df["sender"] == "Showquiz"] if "sender" in df.columns else pd.DataFrame()

    if not quiz_df.empty and "correct" in quiz_df.columns:
        # correct は boolean, 文字列 'true'/'false', 1/0 が混在するため厳密に判定
        correct_series = quiz_df["correct"].astype(str).str.lower().str.strip()
        is_correct = correct_series.isin(["true", "1", "1.0"])
        quiz_acc = is_correct.mean()
    else:
        quiz_acc = np.nan

    quiz_avg_rt = (
        pd.to_numeric(quiz_df["duration"], errors="coerce").mean()
        if not quiz_df.empty and "duration" in quiz_df.columns
        else np.nan
    )

    row_data = {
        "file_name": file.name,
        "participant_id": p_id,
        "age": age,
        "sex": sex,
        "condition": condition,
        "quiz_correct_count": quiz_correct,
        "quiz_total_count": quiz_total,
        "quiz_accuracy": round(quiz_acc, 4) if pd.notna(quiz_acc) else np.nan,
        "quiz_avg_rt_ms": round(quiz_avg_rt, 2) if pd.notna(quiz_avg_rt) else np.nan,
    }

    # 3. 自由記述 (recall2) が複数項目存在する場合（ニュース1、ニュース2の再生記述など）
    if "recall2" in df.columns:
        recalls = df["recall2"].dropna().tolist()
        recalls = [r for r in recalls if str(r).strip() != ""]
        for idx, r_text in enumerate(recalls, start=1):
            row_data[f"recall_text_{idx}"] = r_text

    # 4. アンケート項目・追加質問（q1, q2, purpose, DV項目, comment など）の全自動抽出
    ignore_cols = {
        "sender", "sender_type", "sender_id", "timestamp", "meta", "duration",
        "ended_on", "mean_rt", "time_commit", "time_end", "time_render", "time_run",
        "time_show", "time_switch", "url", "participantID", "age", "sex", "condition",
        "quiz_correct_count", "quiz_total_count", "correct", "recall2", "correctResponse",
        "correct_count", "total_count", "response", "response_action", "seikai"
    }

    for col in df.columns:
        if col in ignore_cols:
            continue

        # 質問・回答系カラムの自動抽出
        vals = df[col].dropna().values
        valid_vals = [v for v in vals if pd.notna(v) and str(v).strip() != ""]
        if len(valid_vals) > 0:
            row_data[col] = valid_vals[0]

    return row_data


if uploaded_files:
    st.success(f"{len(uploaded_files)} 個のファイルが読み込まれました。")

    if st.button("データを結合して処理する"):
        rows = []
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            try:
                row = process_single_file(file)
                rows.append(row)
            except Exception as e:
                st.error(f"エラー ({file.name}): {e}")
            progress_bar.progress((i + 1) / len(uploaded_files))

        summary_df = pd.DataFrame(rows)

        st.subheader("📊 結合データのプレビュー（一部）")
        st.dataframe(summary_df.head(10))

        # CSV化してダウンロードボタンを表示
        csv_data = summary_df.to_csv(index=False, encoding="utf-8-sig").encode(
            "utf-8-sig"
        )
        st.download_button(
            label="💾 まとめCSVファイルをダウンロード",
            data=csv_data,
            file_name="summary_wide_data.csv",
            mime="text/csv",
        )
