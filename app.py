import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="lab.js 統計解析用データ結合ツール", layout="wide"
)

st.title("📊 lab.js 統計解析用データ結合アプリ")
st.write(
    "SPSS / R / Python / Jamovi 等での**統計分析（t検定、分散分析、相関・回帰分析）にそのまま使える完全クリーンなデータセット**（1被験者1行）を作成します。"
)

uploaded_files = st.file_uploader(
    "lab.js の CSV ファイルを選択、またはドラッグ＆ドロップしてください",
    type=["csv"],
    accept_multiple_files=True,
)


def get_valid_numeric_attr(df, col_name, min_val=0, max_val=200):
    """年齢・性別・条件など、妥当な範囲の数値属性を取り出す"""
    if col_name in df.columns:
        vals = df[col_name].dropna().values
        for v in vals:
            try:
                v_num = float(v)
                if min_val <= v_num <= max_val:
                    return int(v_num)
            except Exception:
                pass
    return np.nan


def get_first_valid_text(df, col_name):
    """テキスト属性の最初の有効値を取得"""
    if col_name in df.columns:
        vals = df[col_name].dropna().values
        for v in vals:
            v_str = str(v).strip()
            if pd.notna(v) and v_str != "" and v_str.lower() != "nan":
                return v_str
    return np.nan


def process_single_file_for_stats(file, row_index):
    """
    1被験者分の lab.js CSV から、統計分析に必要な変数（IV, DV, 反応時間, 正誤, 尺度項目）
    のみをシステムノイズ一切なしで綺麗に抽出する関数
    """
    df = pd.read_csv(file)
    row = {}

    # 1. 識別子 & 実験条件 (独立変数: IV) & 被験者属性
    row["id"] = row_index

    p_id = get_first_valid_text(df, "participantID")
    if pd.isna(p_id):
        row["participantID"] = file.name
    else:
        try:
            row["participantID"] = str(int(float(p_id)))
        except (ValueError, TypeError):
            row["participantID"] = str(p_id)

    row["age"] = get_valid_numeric_attr(df, "age", min_val=10, max_val=120)
    row["sex"] = get_valid_numeric_attr(df, "sex", min_val=1, max_val=10)
    row["condition"] = get_valid_numeric_attr(df, "condition", min_val=1, max_val=100)

    # 2. 質問紙尺度・アンケート項目 (従属変数: DV)
    # q1_v1 〜 q1_v5, q2_v6 〜 q2_v10, purpose, comment など
    for col in df.columns:
        c_clean = col.replace(".-", "_").replace(".", "_").replace("-", "_")
        if any(c_clean.startswith(prefix) for prefix in ["q1_", "q2_", "purpose", "comment", "final_consent", "dv_"]):
            val = get_first_valid_text(df, col)
            if pd.notna(val):
                try:
                    val_num = float(val)
                    val = int(val_num) if val_num.is_integer() else val_num
                except Exception:
                    pass
            row[c_clean] = val

    # 3. 自由記述 (Recall)
    if "recall2" in df.columns:
        recalls = df["recall2"].dropna().tolist()
        recalls = [r for r in recalls if str(r).strip().lower() not in ["", "nan"]]
        for idx, r_text in enumerate(recalls, start=1):
            row[f"recall_text_{idx}"] = r_text

    # 4. 全体成績・集計指標
    q_corr = get_valid_numeric_attr(df, "quiz_correct_count", min_val=0, max_val=1000)
    q_tot = get_valid_numeric_attr(df, "quiz_total_count", min_val=1, max_val=1000)
    row["quiz_correct_count"] = q_corr
    row["quiz_total_count"] = q_tot
    if pd.notna(q_corr) and pd.notna(q_tot) and q_tot > 0:
        row["quiz_accuracy"] = round((q_corr / q_tot) * 100, 2)
    else:
        row["quiz_accuracy"] = np.nan

    ldt_corr = get_valid_numeric_attr(df, "correct_count", min_val=0, max_val=1000)
    row["ldt_correct_count"] = ldt_corr
    ldt_rt = get_first_valid_text(df, "mean_rt")
    if pd.notna(ldt_rt):
        try: row["ldt_mean_rt"] = float(ldt_rt)
        except Exception: row["ldt_mean_rt"] = np.nan
    else:
        row["ldt_mean_rt"] = np.nan

    # 5. 各単語・試行レベルの反応時間 (rt_単語名) と 正誤 (corr_単語名: 1/0)
    if "worditem" in df.columns and "duration" in df.columns:
        word_rows = df[df["worditem"].notna() & (df["worditem"] != "")]
        for _, r in word_rows.iterrows():
            w_item = str(r["worditem"]).strip()
            if not w_item or w_item.lower() == "nan":
                continue
            
            w_safe = re.sub(r"[^\w]", "_", w_item)
            
            dur = r.get("duration")
            if pd.notna(dur):
                try:
                    row[f"rt_word_{w_safe}"] = round(float(dur), 1)
                except Exception:
                    pass
            
            corr = str(r.get("correct", "")).lower().strip()
            ans_resp = str(r.get("response", "")).lower().strip()
            sk = str(r.get("seikai", "")).lower().strip()
            if corr in ["true", "1", "1.0"] or (ans_resp != "" and ans_resp == sk):
                row[f"corr_word_{w_safe}"] = 1
            elif corr in ["false", "0", "0.0"]:
                row[f"corr_word_{w_safe}"] = 0

    # 6. クイズ各問の正誤 (quiz_q01 〜 quiz_qN: 1/0)
    if "itemNO" in df.columns and "sender" in df.columns:
        quiz_rows = df[df["sender"].str.contains("quiz", case=False, na=False) & df["itemNO"].notna()]
        for _, r in quiz_rows.iterrows():
            item_no = r.get("itemNO")
            if pd.notna(item_no):
                try:
                    q_num = int(float(item_no))
                    corr = str(r.get("correct", "")).lower().strip()
                    ans_resp = str(r.get("response", "")).lower().strip()
                    sk = str(r.get("seikai", "")).lower().strip()
                    
                    if corr in ["true", "1", "1.0"] or (ans_resp != "" and ans_resp == sk):
                        row[f"quiz_q{q_num:02d}"] = 1
                    elif corr in ["false", "0", "0.0"]:
                        row[f"quiz_q{q_num:02d}"] = 0
                except Exception:
                    pass

    return row


if uploaded_files:
    st.success(f"{len(uploaded_files)} 個のファイルが読み込まれました。")

    if st.button("📊 統計解析用データセットを作成する"):
        rows = []
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            try:
                row = process_single_file_for_stats(file, i + 1)
                rows.append(row)
            except Exception as e:
                st.error(f"エラー ({file.name}): {e}")
            progress_bar.progress((i + 1) / len(uploaded_files))

        summary_df = pd.DataFrame(rows)

        priority_order = [
            "id", "participantID", "age", "sex", "condition",
            "quiz_correct_count", "quiz_total_count", "quiz_accuracy",
            "ldt_correct_count", "ldt_mean_rt"
        ]
        first_cols = [c for c in priority_order if c in summary_df.columns]
        other_cols = [c for c in summary_df.columns if c not in first_cols]
        summary_df = summary_df[first_cols + other_cols]

        st.subheader("📊 統計解析用データセットのプレビュー（ノイズ除去・変数最適化済み）")
        st.write(f"総被験者数 ($N$): **{len(summary_df)}** 人 / 分析変数（カラム数）: **{len(summary_df.columns)}** 変数")
        st.dataframe(summary_df.head(10))

        # CSV化してダウンロード
        csv_data = summary_df.to_csv(index=False, encoding="utf-8-sig").encode(
            "utf-8-sig"
        )
        st.download_button(
            label="💾 統計解析用まとめデータ (summary_stats_data.csv) をダウンロード",
            data=csv_data,
            file_name="summary_stats_data.csv",
            mime="text/csv",
        )
