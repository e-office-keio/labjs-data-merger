import io
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import pykakasi

    _KAKASI = pykakasi.kakasi()
except Exception:
    _KAKASI = None

st.set_page_config(
    page_title="lab.js 汎用データ結合・統計解析用整形ツール", layout="wide"
)

st.title("📊 lab.js 汎用データ結合・統計解析用整形アプリ")
st.write(
    "任意の **lab.js** 実験で収集した複数のCSVファイルを一括アップロードし、"
    "SPSS / R / Python / Jamovi / JASP 等での統計分析にそのまま使える"
    "**ワイド形式（1被験者1行）**のデータセットに自動整形します。"
)

# ==============================================================================
# 1. 定数・標準システム列・ローマ字変換テーブル
# ==============================================================================

# lab.js の時間計測系システム列
LABJS_TIME_COLS: List[str] = [
    "time_run",
    "time_render",
    "time_show",
    "time_end",
    "time_commit",
    "time_switch",
]

# lab.js が標準で全行に出力するシステム列
LABJS_SYSTEM_COLS: Set[str] = {
    "sender",
    "sender_type",
    "sender_id",
    "timestamp",
    "meta",
    "duration",
    "ended_on",
    "response",
    "response_action",
    "correct",
    "url",
    *LABJS_TIME_COLS,
}

# 被験者ID列の候補名（優先順）
PARTICIPANT_ID_CANDIDATES: List[str] = [
    "participantID",
    "participant_id",
    "participantId",
    "participant",
    "sub_id",
    "subject_id",
    "subject",
    "userid",
    "user_id",
    "id",
    "code",
]

# 正解キーが入っている列の候補名
CORRECT_ANSWER_COL_CANDIDATES: List[str] = [
    "seikai",
    "correct_response",
    "correct_answer",
    "answer",
    "ans",
    "target_response",
]

# 試行（トライアル）の識別キーになりやすい列名パターン
TRIAL_KEY_NAME_HINTS: Set[str] = {
    "worditem",
    "word_item",
    "word",
    "itemno",
    "item_no",
    "item_id",
    "itemid",
    "item",
    "stimulus",
    "stim",
    "stim_id",
    "trial_id",
    "trial_no",
    "trial",
    "question_id",
    "q_no",
}

# 自由記述・複数回答になりやすい列名パターン（試行キーから除外するヒント）
FREE_TEXT_NAME_HINTS: Tuple[str, ...] = (
    "recall",
    "free",
    "comment",
    "opinion",
    "thought",
    "impression",
    "feedback",
    "purpose",
    "memo",
    "note",
    "reason",
)

# 既存実験の単語辞書（優先適用）
CUSTOM_ROMAJI_DICT: Dict[str, str] = {
    "じゆう": "jiyuu",
    "いけん": "iken",
    "えんぜつ": "enzetsu",
    "はなしあい": "hanashiai",
    "ばくは": "bakuha",
    "こんらん": "konran",
    "そうどう": "soudou",
    "がいこう": "gaikou",
    "ほうもん": "houmon",
    "みさいる": "misairu",
    "せんかん": "senkan",
    "ぶりょく": "buryoku",
    "からす": "karasu",
    "ちきゅう": "chikyuu",
    "さかな": "sakana",
    "いちご": "ichigo",
    "ぬへちょ": "nuhecho",
    "みらぽん": "mirapon",
    "りんご": "ringo",
    "ぬぽり": "nupori",
    "るちょ": "rucho",
    "まぴそ": "mapiso",
    "けむにん": "kemunin",
    "てぃらま": "tirama",
    "ぱりく": "pariku",
    "とぱけこ": "topakeko",
    "むへろ": "muhero",
    "ふにょま": "funyoma",
}

# 汎用ひらがな→ローマ字変換テーブル（拗音・外来音・濁音・半濁音・五十音）
_KANA_DIGRAPHS: Dict[str, str] = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "じゃ": "ja",  "じゅ": "ju",  "じょ": "jo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "てぃ": "ti",  "でぃ": "di",  "とぅ": "tu",  "どぅ": "du",
    "ふぁ": "fa",  "ふぃ": "fi",  "ふぇ": "fe",  "ふぉ": "fo", "ふゅ": "fyu",
    "うぃ": "wi",  "うぇ": "we",  "うぉ": "wo",
    "ゔぁ": "va",  "ゔぃ": "vi",  "ゔぇ": "ve",  "ゔぉ": "vo", "ゔゅ": "vyu",
    "しぇ": "she", "じぇ": "je",  "ちぇ": "che",
    "つぁ": "tsa", "つぃ": "tsi", "つぇ": "tse", "つぉ": "tso",
}

_KANA_MONOGRAPHS: Dict[str, str] = {
    "あ": "a",  "い": "i",  "う": "u",  "え": "e",  "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "wi", "ゑ": "we", "を": "o",  "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ゔ": "vu",
    "ぁ": "a",  "ぃ": "i",  "ぅ": "u",  "ぇ": "e",  "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ゎ": "wa",
}


def _katakana_to_hiragana(text: str) -> str:
    """カタカナをひらがなに統一"""
    result = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            result.append(chr(code - 0x60))
        else:
            result.append(ch)
    return "".join(result)


def _builtin_kana_to_romaji(text: str) -> str:
    """外部ライブラリなしでも動作する汎用かな→ローマ字変換"""
    hira = _katakana_to_hiragana(text)
    out: List[str] = []
    i = 0
    n = len(hira)
    while i < n:
        if hira[i] == "っ":
            if i + 1 < n:
                next_two = hira[i + 1 : i + 3]
                next_one = hira[i + 1]
                rom = _KANA_DIGRAPHS.get(next_two) or _KANA_MONOGRAPHS.get(next_one, "")
                if rom and rom[0].isalpha():
                    out.append(rom[0])
            i += 1
            continue

        if hira[i] == "ー":
            i += 1
            continue

        if i + 1 < n and hira[i : i + 2] in _KANA_DIGRAPHS:
            out.append(_KANA_DIGRAPHS[hira[i : i + 2]])
            i += 2
            continue

        if hira[i] in _KANA_MONOGRAPHS:
            out.append(_KANA_MONOGRAPHS[hira[i]])
            i += 1
            continue

        out.append(hira[i])
        i += 1

    return "".join(out)


def to_ascii_var_name(text: str, use_romaji: bool = True) -> str:
    """
    任意の文字列（列名や刺激語）を統計ソフト対応のクリーンな変数名に変換する
    """
    text_clean = unicodedata.normalize("NFKC", str(text)).strip()
    if not text_clean:
        return "empty"

    if not use_romaji:
        cleaned = re.sub(r"[\.\-\s/\\:;,]+", "_", text_clean).strip("_")
        return cleaned if cleaned else "var"

    if text_clean in CUSTOM_ROMAJI_DICT:
        return CUSTOM_ROMAJI_DICT[text_clean]

    hira_check = _katakana_to_hiragana(text_clean)
    if all((0x3041 <= ord(c) <= 0x3096) or c == "ー" for c in hira_check):
        romaji_kana = _builtin_kana_to_romaji(text_clean)
        if romaji_kana.isascii() and romaji_kana.isalnum():
            return romaji_kana.lower()

    converted = text_clean
    if not converted.isascii() and _KAKASI is not None:
        try:
            items = _KAKASI.convert(converted)
            converted = "".join(
                item.get("hepburn", item.get("orig", "")) for item in items
            )
        except Exception:
            pass

    if not converted.isascii():
        converted = _builtin_kana_to_romaji(converted)

    converted = re.sub(r"[\.\-\s/\\:;,]+", "_", converted)
    safe_name = re.sub(r"[^0-9a-zA-Z_]", "", converted)
    safe_name = re.sub(r"_+", "_", safe_name).strip("_")

    if not safe_name:
        safe_name = f"item_{abs(hash(text_clean)) % 10000}"
    return safe_name


# ==============================================================================
# 2. ユーティリティ・自動解析関数
# ==============================================================================


def is_system_col(col_name: str) -> bool:
    """lab.jsの標準システム列かどうかを判定（実験者独自の `time_video` 等は除外しない）"""
    c = str(col_name).strip()
    if c in LABJS_SYSTEM_COLS:
        return True
    if c.startswith("meta.") or c.startswith("meta_") or c.startswith("Unnamed:"):
        return True
    return False


def get_valid_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """列から空文字や 'nan' を除いた有効な値の Series を返す"""
    if col_name not in df.columns:
        return pd.Series([], dtype=object)
    s = df[col_name].dropna()
    if s.empty:
        return s
    mask = s.astype(str).str.strip().str.lower().isin(["", "nan", "none", "null"])
    return s[~mask]


def parse_scalar_value(val):
    """数値に変換できる文字列は int または float に変換する"""
    if pd.isna(val):
        return np.nan
    v_str = str(val).strip()
    if v_str == "" or v_str.lower() in ["nan", "none", "null"]:
        return np.nan
    try:
        v_num = float(v_str)
        if np.isfinite(v_num):
            return int(v_num) if v_num.is_integer() else v_num
    except (ValueError, TypeError):
        pass
    return v_str


@st.cache_data(show_spinner=False)
def analyze_uploaded_files(file_bytes_list: List[Tuple[str, bytes]]) -> Dict:
    """
    アップロードされた全CSVファイルをスキャンし、各列および画面（sender）の性質を自動分類する
    """
    all_cols: List[str] = []
    col_set: Set[str] = set()
    dfs: List[Tuple[str, pd.DataFrame]] = []
    sender_names: List[str] = []
    sender_set: Set[str] = set()

    for fname, bdata in file_bytes_list:
        try:
            df = pd.read_csv(io.BytesIO(bdata))
            dfs.append((fname, df))
            for c in df.columns:
                if c not in col_set:
                    col_set.add(c)
                    all_cols.append(c)
            if "sender" in df.columns:
                for s_val in get_valid_series(df, "sender").astype(str).str.strip():
                    if s_val not in sender_set:
                        sender_set.add(s_val)
                        sender_names.append(s_val)
        except Exception:
            pass

    system_cols: List[str] = []
    single_val_cols: List[str] = []
    multi_val_cols: List[str] = []
    trial_key_cols: List[str] = []
    empty_cols: List[str] = []
    present_time_cols: List[str] = [c for c in LABJS_TIME_COLS if c in col_set]

    # 被験者ID列の自動検出
    detected_id_col = "（ファイル名を使用）"
    for cand in PARTICIPANT_ID_CANDIDATES:
        if cand in col_set:
            has_val = any(not get_valid_series(df, cand).empty for _, df in dfs)
            if has_val:
                detected_id_col = cand
                break

    # 正解列の自動検出
    detected_answer_col = "（なし / correct列のみ使用）"
    for cand in CORRECT_ANSWER_COL_CANDIDATES:
        if cand in col_set:
            has_val = any(not get_valid_series(df, cand).empty for _, df in dfs)
            if has_val:
                detected_answer_col = cand
                break

    for col in all_cols:
        if is_system_col(col):
            system_cols.append(col)
            continue
        if col == detected_answer_col:
            system_cols.append(col)
            continue

        max_valid_rows = 0
        max_distinct_vals = 0
        co_occurs_with_rt_or_resp = False

        for _, df in dfs:
            valid_s = get_valid_series(df, col)
            n_rows = len(valid_s)
            if n_rows > 0:
                max_valid_rows = max(max_valid_rows, n_rows)
                n_unique = valid_s.astype(str).str.strip().nunique()
                max_distinct_vals = max(max_distinct_vals, n_unique)

                if n_rows > 1 and n_unique > 1:
                    sub_df = df.loc[valid_s.index]
                    for sys_c in ["duration", "response", "correct"]:
                        if sys_c in sub_df.columns and sub_df[sys_c].notna().any():
                            co_occurs_with_rt_or_resp = True
                            break

        if max_valid_rows == 0:
            empty_cols.append(col)
            continue

        col_lower = col.lower().strip()
        is_trial_hint = col_lower in TRIAL_KEY_NAME_HINTS or any(
            h == col_lower for h in TRIAL_KEY_NAME_HINTS
        )
        is_free_text_hint = any(h in col_lower for h in FREE_TEXT_NAME_HINTS)

        if max_valid_rows > 1 and max_distinct_vals > 1:
            if is_trial_hint or (co_occurs_with_rt_or_resp and not is_free_text_hint):
                trial_key_cols.append(col)
            else:
                multi_val_cols.append(col)
        else:
            single_val_cols.append(col)

    return {
        "all_cols": all_cols,
        "system_cols": system_cols,
        "empty_cols": empty_cols,
        "present_time_cols": present_time_cols,
        "sender_names": sender_names,
        "detected_id_col": detected_id_col,
        "detected_answer_col": detected_answer_col,
        "single_val_cols": single_val_cols,
        "multi_val_cols": multi_val_cols,
        "trial_key_cols": trial_key_cols,
    }


def evaluate_trial_correctness(
    row_series: pd.Series, answer_col: Optional[str]
) -> Optional[int]:
    """
    1試行の正誤を 1 (正解) / 0 (不正解) / None (判定不能) で返す。
    """
    corr_raw = str(row_series.get("correct", "")).lower().strip()
    resp_raw = str(row_series.get("response", "")).lower().strip()
    ans_raw = (
        str(row_series.get(answer_col, "")).lower().strip()
        if answer_col and answer_col in row_series
        else ""
    )

    if corr_raw in ["true", "1", "1.0"]:
        return 1
    if ans_raw not in ["", "nan", "none"] and resp_raw not in ["", "nan", "none"]:
        if resp_raw == ans_raw:
            return 1
        else:
            return 0
    if corr_raw in ["false", "0", "0.0"]:
        return 0
    return None


def extract_single_value(
    df: pd.DataFrame, col_name: str, strategy: str = "last_non_zero"
):
    """
    1被験者につき1つの値を取り出す。
    """
    valid_s = get_valid_series(df, col_name)
    if valid_s.empty:
        return np.nan

    parsed_vals = [parse_scalar_value(v) for v in valid_s.tolist()]
    parsed_vals = [v for v in parsed_vals if pd.notna(v)]
    if not parsed_vals:
        return np.nan

    if strategy == "first":
        return parsed_vals[0]
    if strategy == "last":
        return parsed_vals[-1]

    non_zero_vals = [v for v in parsed_vals if v != 0 and v != "0"]
    if non_zero_vals:
        return non_zero_vals[-1]
    return parsed_vals[-1]


# ==============================================================================
# 3. 1ファイル（1被験者）分のデータ抽出エンジン
# ==============================================================================


def process_participant_df(
    df: pd.DataFrame,
    file_name: str,
    row_index: int,
    id_col: str,
    single_cols: List[str],
    multi_cols: List[str],
    trial_key_cols: List[str],
    answer_col: Optional[str],
    use_romaji: bool,
    single_strategy: str,
    include_trial_rt: bool,
    include_trial_corr: bool,
    include_trial_resp: bool,
    trial_time_cols: List[str],
    calc_task_summary: bool,
    include_file_name: bool,
    include_overall_timing: bool,
    selected_senders_for_time: List[str],
    sender_time_metrics: List[str],
    extract_sender_responses: bool = True,
) -> Dict:
    row: Dict = {}

    # 1. 通し番号 & ファイル名 & 被験者ID
    row["id"] = row_index
    if include_file_name:
        row["file_name"] = file_name

    if id_col and id_col != "（ファイル名を使用）" and id_col in df.columns:
        p_id = extract_single_value(df, id_col, strategy=single_strategy)
        if pd.isna(p_id):
            row["participantID"] = file_name
        else:
            row["participantID"] = str(p_id)
    else:
        row["participantID"] = file_name

    # 1b. 実験全体の時間データ（開始・終了時刻、総所要時間、最終行の time_*）
    if include_overall_timing:
        ts_series = get_valid_series(df, "timestamp")
        if not ts_series.empty:
            row["start_timestamp"] = str(ts_series.iloc[0])
            row["end_timestamp"] = str(ts_series.iloc[-1])

        # 総所要時間 (ms / sec): time_end または time_commit の最大値、なければ duration の合計
        total_ms = None
        for t_col in ["time_end", "time_commit", "time_switch"]:
            if t_col in df.columns:
                num_s = pd.to_numeric(df[t_col], errors="coerce").dropna()
                if not num_s.empty:
                    total_ms = float(num_s.max())
                    break
        if total_ms is None and "duration" in df.columns:
            num_d = pd.to_numeric(df["duration"], errors="coerce").dropna()
            if not num_d.empty:
                total_ms = float(num_d.sum())

        if total_ms is not None:
            row["total_duration_ms"] = round(total_ms, 1)
            row["total_duration_sec"] = round(total_ms / 1000.0, 2)

    # 2. 単一値列（アンケート・属性・実験条件・既存集計値）の抽出
    for col in single_cols:
        if col == id_col:
            continue
        var_name = to_ascii_var_name(col, use_romaji=use_romaji)
        if var_name.lower() == "participantid" and "participantID" in row:
            continue
        row[var_name] = extract_single_value(df, col, strategy=single_strategy)

    # 既存CSVに quiz_correct_count と quiz_total_count があれば quiz_accuracy も自動補完
    if (
        "quiz_correct_count" in row
        and "quiz_total_count" in row
        and "quiz_accuracy" not in row
    ):
        qc = row.get("quiz_correct_count")
        qt = row.get("quiz_total_count")
        if (
            isinstance(qc, (int, float))
            and isinstance(qt, (int, float))
            and pd.notna(qc)
            and pd.notna(qt)
            and qt > 0
        ):
            row["quiz_accuracy"] = round((qc / qt) * 100, 2)

    # 2b. 指定された画面（sender）ごとの滞在時間 (duration) や time_* の抽出
    if selected_senders_for_time and "sender" in df.columns and sender_time_metrics:
        for s_name in selected_senders_for_time:
            s_rows = df[df["sender"].astype(str).str.strip() == s_name]
            if s_rows.empty:
                continue
            s_clean = to_ascii_var_name(s_name, use_romaji=use_romaji)
            for m_col in sender_time_metrics:
                if m_col not in s_rows.columns:
                    continue
                num_vals = pd.to_numeric(s_rows[m_col], errors="coerce").dropna()
                if num_vals.empty:
                    continue
                if len(num_vals) == 1:
                    row[f"{m_col}_{s_clean}"] = round(float(num_vals.iloc[0]), 1)
                else:
                    # 同じ sender 名が複数行ある場合（ループ等）は合計(duration)または最終値(time_*)と平均を出力
                    if m_col == "duration":
                        row[f"duration_sum_{s_clean}"] = round(float(num_vals.sum()), 1)
                        row[f"duration_mean_{s_clean}"] = round(float(num_vals.mean()), 1)
                    else:
                        row[f"{m_col}_first_{s_clean}"] = round(float(num_vals.iloc[0]), 1)
                        row[f"{m_col}_last_{s_clean}"] = round(float(num_vals.iloc[-1]), 1)

    # 3. 複数回答列（自由記述・再生課題など）の連番展開 (`列名_1`, `列名_2`...)
    for col in multi_cols:
        if col not in df.columns:
            continue
        valid_s = get_valid_series(df, col)
        base_name = to_ascii_var_name(col, use_romaji=use_romaji)
        for idx, val in enumerate(valid_s.tolist(), start=1):
            row[f"{base_name}_{idx}"] = parse_scalar_value(val)

    # 4. 課題（トライアル）キー列ごとの横展開 ＆ 課題集計
    used_row_indices: Set[int] = set()

    def _key_sort_priority(k: str) -> int:
        if k not in df.columns:
            return 99
        s = get_valid_series(df, k)
        if s.empty:
            return 99
        all_num = all(
            isinstance(parse_scalar_value(v), (int, float)) for v in s.head(10)
        )
        return 1 if all_num else 0

    sorted_trial_keys = sorted(trial_key_cols, key=_key_sort_priority)

    for key_col in sorted_trial_keys:
        if key_col not in df.columns:
            continue

        valid_s = get_valid_series(df, key_col)
        if valid_s.empty:
            continue

        remaining_indices = [
            idx for idx in valid_s.index if idx not in used_row_indices
        ]
        if not remaining_indices:
            continue

        trial_rows = df.loc[remaining_indices]
        used_row_indices.update(remaining_indices)

        key_col_clean = to_ascii_var_name(key_col, use_romaji=use_romaji)
        has_quiz_sender = (
            "sender" in trial_rows.columns
            and trial_rows["sender"]
            .astype(str)
            .str.contains("quiz", case=False, na=False)
            .any()
        )
        if key_col.lower() == "worditem":
            prefix_label = "word"
        elif has_quiz_sender and key_col.lower() in ["itemno", "item_no", "item"]:
            prefix_label = "quiz"
        else:
            prefix_label = key_col_clean

        task_rts: List[float] = []
        task_correct_rts: List[float] = []
        task_corrs: List[int] = []

        for _, r in trial_rows.iterrows():
            raw_key_val = r[key_col]
            parsed_key = parse_scalar_value(raw_key_val)

            if isinstance(parsed_key, int):
                item_suffix = (
                    f"q{parsed_key:02d}"
                    if prefix_label == "quiz"
                    else f"{parsed_key:02d}"
                )
            else:
                item_suffix = to_ascii_var_name(
                    str(raw_key_val), use_romaji=use_romaji
                )

            # (a) 反応時間 (duration)
            rt_val = None
            if "duration" in r and pd.notna(r["duration"]):
                try:
                    rt_val = round(float(r["duration"]), 1)
                    task_rts.append(rt_val)
                except (ValueError, TypeError):
                    pass

            # (b) 正誤 (correct / response == answer_col)
            corr_val = evaluate_trial_correctness(r, answer_col)
            if corr_val is not None:
                task_corrs.append(corr_val)
                if corr_val == 1 and rt_val is not None:
                    task_correct_rts.append(rt_val)

            # (c) 個別試行カラムへの書き出し
            if prefix_label == "quiz" and key_col.lower() in [
                "itemno",
                "item_no",
                "item",
            ]:
                if include_trial_corr and corr_val is not None:
                    row[f"quiz_{item_suffix}"] = corr_val
                if include_trial_rt and rt_val is not None:
                    row[f"rt_quiz_{item_suffix}"] = rt_val
                if include_trial_resp and "response" in r and pd.notna(r["response"]):
                    row[f"resp_quiz_{item_suffix}"] = parse_scalar_value(r["response"])
            else:
                if include_trial_rt and rt_val is not None:
                    row[f"rt_{prefix_label}_{item_suffix}"] = rt_val
                if include_trial_corr and corr_val is not None:
                    row[f"corr_{prefix_label}_{item_suffix}"] = corr_val
                if include_trial_resp and "response" in r and pd.notna(r["response"]):
                    row[f"resp_{prefix_label}_{item_suffix}"] = parse_scalar_value(
                        r["response"]
                    )

            # (d) 各試行の time_* データ（選択された場合）
            for t_col in trial_time_cols:
                if t_col in r and pd.notna(r[t_col]):
                    try:
                        row[f"{t_col}_{prefix_label}_{item_suffix}"] = round(
                            float(r[t_col]), 1
                        )
                    except (ValueError, TypeError):
                        pass

        # (e) 課題全体の自動集計指標（正答数・総数・正答率・平均RT）
        if calc_task_summary:
            if task_corrs:
                c_cnt = int(sum(task_corrs))
                t_cnt = len(task_corrs)
                acc = round((c_cnt / t_cnt) * 100, 2) if t_cnt > 0 else np.nan
                row.setdefault(f"{prefix_label}_calc_correct_count", c_cnt)
                row.setdefault(f"{prefix_label}_calc_total_count", t_cnt)
                row.setdefault(f"{prefix_label}_calc_accuracy", acc)
            if task_rts:
                row.setdefault(
                    f"{prefix_label}_calc_mean_rt", round(float(np.mean(task_rts)), 2)
                )
            if task_correct_rts:
                row.setdefault(
                    f"{prefix_label}_calc_correct_mean_rt",
                    round(float(np.mean(task_correct_rts)), 2),
                )

    # 5. 試行キー列を持たない `response` 行（canvas.Screen のボタンクリック回答など）を `sender` 名で自動抽出
    if extract_sender_responses and "response" in df.columns and "sender" in df.columns:
        resp_s = get_valid_series(df, "response")
        rem_resp_indices = [
            idx for idx in resp_s.index if idx not in used_row_indices
        ]
        if rem_resp_indices:
            resp_rows = df.loc[rem_resp_indices]
            # 同じ sender 名が複数回出現するかカウント
            sender_counts: Dict[str, int] = {}
            for _, r in resp_rows.iterrows():
                s_raw = str(r.get("sender", "screen")).strip()
                s_clean = to_ascii_var_name(s_raw, use_romaji=use_romaji)
                sender_counts[s_clean] = sender_counts.get(s_clean, 0) + 1

            sender_seen: Dict[str, int] = {}
            for _, r in resp_rows.iterrows():
                s_raw = str(r.get("sender", "screen")).strip()
                s_clean = to_ascii_var_name(s_raw, use_romaji=use_romaji)
                sender_seen[s_clean] = sender_seen.get(s_clean, 0) + 1

                if sender_counts[s_clean] > 1:
                    var_label = f"{s_clean}_{sender_seen[s_clean]:02d}"
                else:
                    var_label = s_clean

                # 回答値
                row[var_label] = parse_scalar_value(r["response"])

                # 反応時間 (duration)
                if include_trial_rt and "duration" in r and pd.notna(r["duration"]):
                    try:
                        row[f"rt_{var_label}"] = round(float(r["duration"]), 1)
                    except (ValueError, TypeError):
                        pass

                # 正誤 (存在する場合)
                corr_val = evaluate_trial_correctness(r, answer_col)
                if include_trial_corr and corr_val is not None:
                    row[f"corr_{var_label}"] = corr_val

                # 追加の time_* 列
                for t_col in trial_time_cols:
                    if t_col in r and pd.notna(r[t_col]):
                        try:
                            row[f"{t_col}_{var_label}"] = round(float(r[t_col]), 1)
                        except (ValueError, TypeError):
                            pass

    # 6. HTMLフォームで name 属性が空欄（Unnamed: * 列）だった入力値の救出（その行の sender 名を変数名として補完）
    for col in df.columns:
        if str(col).startswith("Unnamed:"):
            valid_u = get_valid_series(df, col)
            for idx, val in valid_u.items():
                s_raw = (
                    str(df.loc[idx, "sender"]).strip()
                    if "sender" in df.columns and pd.notna(df.loc[idx, "sender"])
                    else col
                )
                s_clean = to_ascii_var_name(s_raw, use_romaji=use_romaji)
                row.setdefault(s_clean, parse_scalar_value(val))

    return row


def _natural_col_sort_key(col_name: str) -> Tuple:
    """
    ランダム呈示された刺激列（SNS2, SNS3, SNS1... 等）を番号順（SNS1, SNS2, SNS3...）に並べ、
    かつ本回答列の後ろに rt_* / corr_* / time_* 列がまとまるようにソートキーを生成する
    """
    if col_name.startswith("rt_"):
        group = 1
    elif col_name.startswith("corr_"):
        group = 2
    elif any(col_name.startswith(f"{tc}_") for tc in LABJS_TIME_COLS) or col_name.startswith("duration_"):
        group = 3
    else:
        group = 0

    parts = [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", col_name)
    ]
    return (group, parts)


# ==============================================================================
# 4. メイン UI
# ==============================================================================

uploaded_files = st.file_uploader(
    "📁 lab.js の CSV ファイルを選択、またはドラッグ＆ドロップしてください（複数選択可）",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} 個のCSVファイルが読み込まれました。")

    file_bytes_list = [(f.name, f.getvalue()) for f in uploaded_files]
    analysis = analyze_uploaded_files(file_bytes_list)
    all_non_empty_cols = [
        c for c in analysis["all_cols"] if c not in analysis["empty_cols"]
    ]

    with st.expander(
        "⚙️ 抽出・展開ルールの確認とカスタマイズ（自動検出済み）", expanded=True
    ):
        st.caption(
            "アップロードされたCSVの列構造を自動解析しました。そのまま下の作成ボタンを押すか、必要に応じて設定を調整してください。"
        )

        col_left, col_right = st.columns(2)

        with col_left:
            id_options = ["（ファイル名を使用）"] + all_non_empty_cols
            default_id_idx = (
                id_options.index(analysis["detected_id_col"])
                if analysis["detected_id_col"] in id_options
                else 0
            )
            selected_id_col = st.selectbox(
                "🆔 被験者IDとして使用する列",
                options=id_options,
                index=default_id_idx,
                help="被験者IDが記録されている列を選択します。存在しない場合はファイル名が participantID になります。",
            )

            ans_options = ["（なし / correct列のみ使用）"] + all_non_empty_cols
            default_ans_idx = (
                ans_options.index(analysis["detected_answer_col"])
                if analysis["detected_answer_col"] in ans_options
                else 0
            )
            selected_ans_col = st.selectbox(
                "🎯 正解キーが格納されている列（任意）",
                options=ans_options,
                index=default_ans_idx,
                help="lab.js標準の `correct` 列以外に、`response` と比較する正解列（例: `seikai`, `answer`）がある場合に指定します。",
            )

            single_strategy_label = st.selectbox(
                "📌 単一値列で複数行に値がある場合の優先ルール",
                options=[
                    "最後の非ゼロ有効値を優先（推奨：lab.jsのstate更新に対応）",
                    "最初の有効値を取得",
                    "最後の有効値を取得",
                ],
                index=0,
            )
            strategy_map = {
                "最後の非ゼロ有効値を優先（推奨：lab.jsのstate更新に対応）": "last_non_zero",
                "最初の有効値を取得": "first",
                "最後の有効値を取得": "last",
            }
            single_strategy = strategy_map[single_strategy_label]

        with col_right:
            use_romaji = st.checkbox(
                "🔤 日本語の列名・刺激語をローマ字（ASCII英数字）に自動変換する",
                value=True,
                help="SPSS / R / Jamovi 等で文字化けや変数名エラーが起きないよう、ひらがな・カタカナ・漢字をローマ字に変換し記号を `_` に統一します。",
            )
            extract_sender_responses = st.checkbox(
                "🖱️ 画面ごとのクリック・キー回答（`sender` × `response`列）を自動抽出する",
                value=True,
                help="`canvas.Screen` のボタンクリック等で `response` 列に記録された回答（例: `SNS1_reliability`, `SNS1_buy` 等）を画面名ごとに1列ずつ抽出します。",
            )
            include_file_name = st.checkbox(
                "📄 元のCSVファイル名 (`file_name`) 列も出力する",
                value=False,
            )
            include_overall_timing = st.checkbox(
                "🕒 実験全体の開始・終了日時 (`timestamp`) と総所要時間 (`total_duration_sec`) を出力する",
                value=True,
                help="最終行の `time_end` / `time_commit` 等から実験全体の所要時間（ミリ秒・秒）を自動算出し、不良回答スクリーニングに使えます。",
            )
            calc_task_summary = st.checkbox(
                "📈 課題ごとの正答率・平均RT（全体／正答のみ）を自動計算して追加する",
                value=True,
            )

            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                include_trial_rt = st.checkbox("⏱️ 試行ごとのRT", value=True)
            with t_col2:
                include_trial_corr = st.checkbox("⭕ 試行ごとの正誤", value=True)
            with t_col3:
                include_trial_resp = st.checkbox("💬 試行キーの回答値", value=False)

        st.divider()

        selected_single_cols = st.multiselect(
            "📌 単一値として抽出する列（アンケート・属性・実験条件・既存集計スコアなど：1被験者1列）",
            options=all_non_empty_cols,
            default=analysis["single_val_cols"],
            help="1被験者につき1つの値を持つ列です。もし `time_end` 等の最終値だけを1列として取り出したい場合はここに追加することもできます。",
        )

        selected_multi_cols = st.multiselect(
            "📝 連番展開する複数回答列（自由記述の複数回答・再生課題など：`列名_1`, `列名_2`...）",
            options=all_non_empty_cols,
            default=analysis["multi_val_cols"],
            help="1人の被験者が複数行にわたって回答した列（自由再生など）を、`列名_1`, `列名_2`... として横に展開します。",
        )

        selected_trial_keys = st.multiselect(
            "⏱️ 課題（トライアル）の試行キー列（刺激語・問題番号ごとにRTや正誤を横展開）",
            options=all_non_empty_cols,
            default=analysis["trial_key_cols"],
            help="刺激語（例: `worditem`）や問題番号（例: `itemNO`）の列を指定すると、各刺激・各問ごとの反応時間や正誤を横展開します。",
        )

        # 🕒 詳細な time_* / 画面別滞在時間の設定
        st.markdown("##### 🕒 `time_*` データ・画面（`sender`）別の所要時間の詳細抽出")
        show_time_legend = st.checkbox(
            "📖 時間変数（`time_*` / `duration` / `timestamp`）の意味・凡例を表示する",
            value=True,
        )
        if show_time_legend:
            st.info(
                """
**⏱️ lab.js における1画面（1コンポーネント）の時間経過イメージ**
`time_run` ➔ `time_render` ➔ **`time_show` ──( 画面表示中：`duration` )──▶ `time_end`** ➔ `time_commit` ➔ `time_switch`

| 変数名 | 単位 | 意味・タイミング | 主な分析用途 |
| :--- | :---: | :--- | :--- |
| **`duration`** | ms (ミリ秒) | **画面滞在時間・反応時間**（`≈ time_end - time_show`） | **最重要**：課題の反応時間（RT）や、映像・教示・質問紙画面の滞在時間チェック |
| **`timestamp`** | 日時 (ISO) | その画面が終了・記録された実世界の時計の日時 | 実験の実施時間帯の確認、開始〜終了時刻の把握 |
| **`time_run`** | ms (実験開始から) | コンポーネントの内部処理（スクリプト実行）が開始された時点 | プログラム上のコンポーネント起動タイミングの確認 |
| **`time_render`** | ms (実験開始から) | 画面のHTML/Canvasが組み立てられ、ブラウザ描画が要求された時点 | 描画準備にかかった時間の検証 |
| **`time_show`** | ms (実験開始から) | **実際に画面上に刺激が表示された時点**（フレーム描画完了） | 刺激呈示の正確な開始タイミングの確認（`time_show - time_render` で描画遅延を確認可） |
| **`time_end`** | ms (実験開始から) | **応答（キー/クリック）や制限時間により画面が終了した時点** | 各画面の終了タイミング。**最終行の `time_end` ＝ 実験全体の総所要時間** |
| **`time_commit`** | ms (実験開始から) | その画面の回答データが内部データストアに保存（commit）された時点 | データ記録完了タイミングの確認 |
| **`time_switch`** | ms (実験開始から) | 次の画面（コンポーネント）への切り替えが完了した時点 | 画面遷移のオーバーヘッド確認 |

- **本ツールが自動算出する全体指標**: `total_duration_sec` / `total_duration_ms`（実験全体の総所要時間：不良回答スクリーニング用）、`start_timestamp` / `end_timestamp`（実験開始・終了日時）
"""
            )

        time_col_left, time_col_right = st.columns(2)
        with time_col_left:
            selected_senders_for_time = st.multiselect(
                "🖥️ 画面（`sender`）ごとに滞在時間や `time_*` を抽出する画面名（任意）",
                options=analysis["sender_names"],
                default=[],
                help="例えば映像視聴画面や教示画面・アンケート画面の `sender` 名を選択すると、その画面の滞在時間 (`duration`) や `time_*` を被験者ごとに抽出します。",
            )
            sender_time_metrics = st.multiselect(
                "↪ 選択した画面（`sender`）で出力する時間指標",
                options=["duration"] + analysis["present_time_cols"],
                default=(
                    ["duration", "time_show", "time_end"]
                    if "time_show" in analysis["present_time_cols"]
                    else ["duration"]
                ),
            )
        with time_col_right:
            trial_time_cols = st.multiselect(
                "⏱️ 各試行（トライアル）ごとに追加出力する `time_*` 列（任意）",
                options=analysis["present_time_cols"],
                default=[],
                help="試行ごとの `duration`（反応時間）に加えて、各試行の `time_show`, `time_end`, `time_commit` 等も出力したい場合に選択してください。",
            )

    if st.button(
        "📊 統計解析用ワイドデータセット（1被験者1行）を作成する", type="primary"
    ):
        rows = []
        progress_bar = st.progress(0)
        ans_col_arg = (
            None
            if selected_ans_col == "（なし / correct列のみ使用）"
            else selected_ans_col
        )

        for i, (fname, bdata) in enumerate(file_bytes_list):
            try:
                df = pd.read_csv(io.BytesIO(bdata))
                row = process_participant_df(
                    df=df,
                    file_name=fname,
                    row_index=i + 1,
                    id_col=selected_id_col,
                    single_cols=selected_single_cols,
                    multi_cols=selected_multi_cols,
                    trial_key_cols=selected_trial_keys,
                    answer_col=ans_col_arg,
                    use_romaji=use_romaji,
                    single_strategy=single_strategy,
                    include_trial_rt=include_trial_rt,
                    include_trial_corr=include_trial_corr,
                    include_trial_resp=include_trial_resp,
                    trial_time_cols=trial_time_cols,
                    calc_task_summary=calc_task_summary,
                    include_file_name=include_file_name,
                    include_overall_timing=include_overall_timing,
                    selected_senders_for_time=selected_senders_for_time,
                    sender_time_metrics=sender_time_metrics,
                    extract_sender_responses=extract_sender_responses,
                )
                rows.append(row)
            except Exception as e:
                st.error(f"エラー ({fname}): {e}")
            progress_bar.progress((i + 1) / len(file_bytes_list))

        summary_df = pd.DataFrame(rows)

        # 優先度の高い基本列を先頭に整理し、残りの列は番号順（SNS1, SNS2...）に自然ソート
        priority_order = [
            "id",
            "file_name",
            "participantID",
            "age",
            "sex",
            "gender",
            "condition",
            "group",
            "total_duration_sec",
            "total_duration_ms",
            "start_timestamp",
            "end_timestamp",
        ]
        first_cols = [c for c in priority_order if c in summary_df.columns]
        other_cols = sorted(
            [c for c in summary_df.columns if c not in first_cols],
            key=_natural_col_sort_key,
        )
        summary_df = summary_df[first_cols + other_cols]

        st.subheader("📊 データセットプレビュー（1被験者1行・ワイド形式）")
        st.write(
            f"総被験者数 ($N$): **{len(summary_df)}** 人 ／ "
            f"抽出された変数（カラム数）: **{len(summary_df.columns)}** 変数"
        )
        st.dataframe(summary_df, use_container_width=True)

        # CSVダウンロード（UTF-8 BOM付き）
        csv_data = summary_df.to_csv(index=False, encoding="utf-8-sig").encode(
            "utf-8-sig"
        )
        st.download_button(
            label="💾 統計解析用まとめデータ (summary_stats_data.csv) をダウンロード",
            data=csv_data,
            file_name="summary_stats_data.csv",
            mime="text/csv",
            type="primary",
        )

