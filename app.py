import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="lab.js 万能データ結合ツール", layout="wide"
)

st.title("🧪 lab.js 実験データ万能結合アプリ")
st.write(
    "どんな lab.js の実験データファイル（CSV）でも、全被験者・全試行データを自動解析し、1被験者1行（ワイド形式）にすべてまとめてダウンロードできます。"
)

uploaded_files = st.file_uploader(
    "lab.js の CSV ファイルを複数選択、またはドラッグ＆ドロップしてください",
    type=["csv"],
    accept_multiple_files=True,
)


def process_single_file_generic(file, row_index):
    """
    あらゆる lab.js 実験の CSV ファイルを全自動解析し、
    漏れなく1行 (Wide format) に変換する汎用関数
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

    # 3. システム管理用除外カラム
    ignore_cols = {
        "sender_type", "sender_id", "timestamp", "meta", "ended_on", "url",
        "time_commit", "time_end", "time_render", "time_run", "time_show", "time_switch",
        "lessbrgreaterlessbrgreater-approved", "participantID"
    }

    # 各カラムの全自動抽出・ワイド化
    for col in df.columns:
        if col in ignore_cols or col.startswith("Unnamed:"):
            continue

        valid_series = df[col].dropna()
        valid_vals = [v for v in valid_series.values if str(v).strip().lower() not in ["", "nan"]]

        if len(valid_vals) == 0:
            continue

        # A) 1つのファイル内で単一回答（被験者属性・アンケート・単一の集計値など）のカラム
        if len(set(str(v) for v in valid_vals)) == 1 or len(valid_vals) == 1:
            val = valid_vals[0]
            # 数値データのフォーマット
            try:
                if str(val).endswith(".0"):
                    val = int(float(val))
            except Exception:
                pass
            row[col] = val
        else:
            # B) 試行ごとに複数存在するカラム (duration, correct, response, worditem, recall2 等)
            for idx, (r_idx, val) in enumerate(valid_series.items()):
                val_str = str(val).strip()
                if val_str.lower() in ["", "nan"]:
                    continue

                # キーコンテキスト（sender, worditem, itemNO, headline）の抽出
                sender_val = str(df.loc[r_idx, "sender"]).strip() if "sender" in df.columns and pd.notna(df.loc[r_idx, "sender"]) else ""
                word_val = str(df.loc[r_idx, "worditem"]).strip() if "worditem" in df.columns and pd.notna(df.loc[r_idx, "worditem"]) else ""
                item_val = str(df.loc[r_idx, "itemNO"]).strip() if "itemNO" in df.columns and pd.notna(df.loc[r_idx, "itemNO"]) else ""
                hl_val = str(df.loc[r_idx, "headline"]).strip() if "headline" in df.columns and pd.notna(df.loc[r_idx, "headline"]) else ""

                sub_key_parts = []
                if sender_val and sender_val.lower() != "nan":
                    sub_key_parts.append(sender_val)
                if word_val and word_val.lower() != "nan":
                    sub_key_parts.append(word_val)
                elif item_val and item_val.lower() != "nan":
                    try:
                        item_num = int(float(item_val))
                        sub_key_parts.append(f"q{item_num:02d}")
                    except Exception:
                        sub_key_parts.append(f"item_{item_val}")
                elif hl_val and hl_val.lower() != "nan":
                    sub_key_parts.append(hl_val[:10])

                if len(sub_key_parts) > 0:
                    sub_key = "_".join(sub_key_parts)
                    wide_col_name = f"{col}_{sub_key}"
                else:
                    wide_col_name = f"{col}_{idx + 1}"

                # boolean 型正誤判定の修正 (正答判定が 'false' の場合に True になるバグを防止)
                if col == "correct":
                    is_c = str(val).lower().strip() in ["true", "1", "1.0"]
                    row[wide_col_name] = 1 if is_c else 0
                else:
                    row[wide_col_name] = val

    return row


if uploaded_files:
    st.success(f"{len(uploaded_files)} 個のファイルが読み込まれました。")

    if st.button("データを結合して処理する"):
        rows = []
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            try:
                row = process_single_file_generic(file, i + 1)
                rows.append(row)
            except Exception as e:
                st.error(f"エラー ({file.name}): {e}")
            progress_bar.progress((i + 1) / len(uploaded_files))

        summary_df = pd.DataFrame(rows)

        # 列の整列 (id, participantID, file_name, start_time を先頭に)
        priority_cols = ["id", "participantID", "file_name", "start_time", "age", "sex", "condition"]
        first_cols = [c for c in priority_cols if c in summary_df.columns]
        other_cols = [c for c in summary_df.columns if c not in first_cols]
        summary_df = summary_df[first_cols + other_cols]

        st.subheader("📊 結合データのプレビュー（一部）")
        st.write(f"総被験者数: {len(summary_df)} 人 / 抽出カラム数: {len(summary_df.columns)} 列")
        st.dataframe(summary_df.head(10))

        # まとめCSVのダウンロード
        csv_data = summary_df.to_csv(index=False, encoding="utf-8-sig").encode(
            "utf-8-sig"
        )
        st.download_button(
            label="💾 まとめCSVファイルをダウンロード",
            data=csv_data,
            file_name="summary_wide_data.csv",
            mime="text/csv",
        )
