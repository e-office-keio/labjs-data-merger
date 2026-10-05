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


def process_single_file(file):
    """1つのファイル（1被験者分）を1行のデータに変換する関数"""
    df = pd.read_csv(file)

    # 被験者情報
    p_id = (
        df["participantID"].dropna().values[0]
        if "participantID" in df.columns and len(df["participantID"].dropna()) > 0
        else file.name
    )
    age = (
        df["age"].dropna().values[0]
        if "age" in df.columns and len(df["age"].dropna()) > 0
        else np.nan
    )
    sex = (
        df["sex"].dropna().values[0]
        if "sex" in df.columns and len(df["sex"].dropna()) > 0
        else np.nan
    )
    condition = (
        df["condition"].dropna().values[0]
        if "condition" in df.columns and len(df["condition"].dropna()) > 0
        else np.nan
    )

    # クイズ・集計指標
    quiz_correct = (
        df["quiz_correct_count"].dropna().values[0]
        if "quiz_correct_count" in df.columns
        and len(df["quiz_correct_count"].dropna()) > 0
        else np.nan
    )
    quiz_total = (
        df["quiz_total_count"].dropna().values[0]
        if "quiz_total_count" in df.columns
        and len(df["quiz_total_count"].dropna()) > 0
        else np.nan
    )

    # クイズ正答率・平均反応時間計算
    quiz_df = (
        df[df["sender"] == "Showquiz"] if "sender" in df.columns else pd.DataFrame()
    )
    quiz_acc = (
        quiz_df["correct"].astype(bool).mean()
        if not quiz_df.empty and "correct" in quiz_df.columns
        else np.nan
    )
    quiz_avg_rt = (
        quiz_df["duration"].mean()
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
        "quiz_accuracy": quiz_acc,
        "quiz_avg_rt_ms": quiz_avg_rt,
    }

    # アンケート等の追加項目（q1., q2., purposeなど）を抽出
    for col in df.columns:
        if col.startswith("q1.") or col.startswith("q2.") or col.startswith("purpose"):
            vals = df[col].dropna().values
            if len(vals) > 0:
                row_data[col] = vals[0]

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
