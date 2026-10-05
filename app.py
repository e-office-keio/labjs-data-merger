import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="lab.js 実験データ結合ツール", layout="wide"
)

st.title("🧪 lab.js 実験データ結合アプリ")
st.write(
    "複数の lab.js CSV ファイルをアップロードし、分析に必要なデータだけをすっきり整理して1被験者1行（ワイド形式）に結合します。"
)

# ファイルアップローダー
uploaded_files = st.file_uploader(
    "lab.js の CSV ファイルを選択、またはドラッグ＆ドロップしてください",
    type=["csv"],
    accept_multiple_files=True,
)

# サイドバーまたは設定項目：不要列のフィルタリング
st.sidebar.header("⚙️ 抽出オプション")
filter_redundant = st.sidebar.checkbox(
    "分析不要な内部データ（sender名, correctResponse, アクションログ等）を除外する",
    value=True
)


def process_single_file_clean(file, row_index):
    """
    lab.js CSV から分析に必要な主要データのみを綺麗に抽出する汎用関数
    """
    df = pd.read_csv(file)
    row = {}

    row["id"] = row_index
    row["file_name"] = file.name

    # 1. 被験者ID
    p_id = np.nan
    if "participantID" in df.columns:
        valid_ids = df["participantID"].dropna().values
        for v in valid_ids:
            v_str = str(v).strip()
            if v_str != "" and v_str.lower() != "nan":
                p_id = v_str
                break

    if pd.isna(p_id) or p_id == "":
        row["participantID"] = file.name
    else:
        try:
            row["participantID"] = str(int(float(p_id)))
        except (ValueError, TypeError):
            row["participantID"] = str(p_id)

    # 2. 実験開始日時 (timestamp の最古値)
    if "timestamp" in df.columns:
        valid_ts = df["timestamp"].dropna().values
        if len(valid_ts) > 0:
            try:
                dt = pd.to_datetime(valid_ts[0])
                row["start_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                row["start_time"] = str(valid_ts[0])

    # 完全除外するシステム・ログ系カラム
    always_ignore = {
        "sender_type", "sender_id", "timestamp", "meta", "ended_on", "url",
        "time_commit", "time_end", "time_render", "time_run", "time_show", "time_switch",
        "lessbrgreaterlessbrgreater-approved", "participantID"
    }

    # 各カラムの処理
    for col in df.columns:
        if col in always_ignore or col.startswith("Unnamed:"):
            continue

        valid_series = df[col].dropna()
        valid_vals = [v for v in valid_series.values if str(v).strip().lower() not in ["", "nan"]]

        if len(valid_vals) == 0:
            continue

        # A) 単一被験者属性・集計値・アンケート質問（q1.-v1, age, sex, condition, purpose 等）
        if len(set(str(v) for v in valid_vals)) == 1 or len(valid_vals) == 1:
            val = valid_vals[0]
            try:
                if str(val).endswith(".0"):
                    val = int(float(val))
            except Exception:
                pass
            row[col] = val
        else:
            # B) 試行ごとの繰り返しデータ (duration, correct, response, recall2 等)
            for idx, (r_idx, val) in enumerate(valid_series.items()):
                val_str = str(val).strip()
                if val_str.lower() in ["", "nan"]:
                    continue

                # 識別コンテキスト (worditem, itemNO, headline)
                sender_val = str(df.loc[r_idx, "sender"]).strip() if "sender" in df.columns and pd.notna(df.loc[r_idx, "sender"]) else ""
                word_val = str(df.loc[r_idx, "worditem"]).strip() if "worditem" in df.columns and pd.notna(df.loc[r_idx, "worditem"]) else ""
                item_val = str(df.loc[r_idx, "itemNO"]).strip() if "itemNO" in df.columns and pd.notna(df.loc[r_idx, "itemNO"]) else ""
                hl_val = str(df.loc[r_idx, "headline"]).strip() if "headline" in df.columns and pd.notna(df.loc[r_idx, "headline"]) else ""

                sub_key_parts = []
                if word_val and word_val.lower() != "nan":
                    sub_key_parts.append(word_val)
                elif item_val and item_val.lower() != "nan":
                    try:
                        item_num = int(float(item_val))
                        sub_key_parts.append(f"q{item_num:02d}")
                    except Exception:
                        sub_key_parts.append(f"item_{item_val}")
                elif hl_val and hl_val.lower() != "nan":
                    sub_key_parts.append(hl_val[:8])
                elif sender_val and sender_val.lower() != "nan":
                    sub_key_parts.append(sender_val)

                if len(sub_key_parts) > 0:
                    sub_key = "_".join(sub_key_parts)
                    wide_col_name = f"{col}_{sub_key}"
                else:
                    wide_col_name = f"{col}_{idx + 1}"

                # boolean 型の正誤判定修正
                if col == "correct":
                    is_c = str(val).lower().strip() in ["true", "1", "1.0"]
                    row[wide_col_name] = 1 if is_c else 0
                else:
                    row[wide_col_name] = val

    return row


def filter_clean_columns(df):
    """分析に不要な冗長列（sender, correctResponse, アクション詳細等）を除外"""
    clean_cols = []
    for col in df.columns:
        c_lower = col.lower()
        # 除外キーワード
        if (
            c_lower.startswith("sender") or
            c_lower.startswith("correctresponse") or
            c_lower.startswith("response_action") or
            c_lower.startswith("seikai") or
            c_lower.startswith("quizitem") or
            c_lower.startswith("worditem") or
            c_lower.startswith("newsitem")
        ):
            continue
        clean_cols.append(col)
    return df[clean_cols]


if uploaded_files:
    st.success(f"{len(uploaded_files)} 個のファイルが読み込まれました。")

    if st.button("データを結合して処理する"):
        rows = []
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            try:
                row = process_single_file_clean(file, i + 1)
                rows.append(row)
            except Exception as e:
                st.error(f"エラー ({file.name}): {e}")
            progress_bar.progress((i + 1) / len(uploaded_files))

        summary_df = pd.DataFrame(rows)

        # 1. カラムの先頭整理
        priority_cols = ["id", "participantID", "file_name", "start_time", "age", "sex", "condition"]
        first_cols = [c for c in priority_cols if c in summary_df.columns]
        other_cols = [c for c in summary_df.columns if c not in first_cols]
        summary_df = summary_df[first_cols + other_cols]

        # 2. 不要データの除去（フィルタ設定がONの場合）
        if filter_redundant:
            summary_df = filter_clean_columns(summary_df)

        st.subheader("📊 結合データのプレビュー（一部）")
        st.write(f"総被験者数: {len(summary_df)} 人 / 抽出データ項目数: {len(summary_df.columns)} 列")
        st.dataframe(summary_df.head(10))

        # まとめCSVダウンロード
        csv_data = summary_df.to_csv(index=False, encoding="utf-8-sig").encode(
            "utf-8-sig"
        )
        st.download_button(
            label="💾 まとめCSVファイルをダウンロード",
            data=csv_data,
            file_name="summary_wide_data.csv",
            mime="text/csv",
        )
