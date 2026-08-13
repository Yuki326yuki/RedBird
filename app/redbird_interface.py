from __future__ import annotations

import csv
import sys
from io import StringIO
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sdg_extract.chatbot import DEFAULT_CHATBOT_CONFIG, answer_question  # noqa: E402
from sdg_extract.config import load_config, resolve_path  # noqa: E402
from sdg_extract.database import database_is_stale, database_summary, distinct_values, query_policies  # noqa: E402


DEFAULT_DB = PROJECT_ROOT / "data/db/redbird_sdg13_v2.sqlite"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/sdg13_china_us_gpt56_luna_v2.yaml"
DEFAULT_PROMPT = PROJECT_ROOT / "prompts/chatbot/policy_qa.md"


st.set_page_config(page_title="Red Bird 政策数据库", layout="wide")
st.title("Red Bird 政策数据库")


def load_chatbot_config(config_path: Path, db_path: Path) -> tuple[dict, Path]:
    if not config_path.exists():
        config = dict(DEFAULT_CHATBOT_CONFIG)
        config["db_path"] = str(db_path)
        return config, DEFAULT_PROMPT
    full_config = load_config(config_path)
    chatbot_config = dict(full_config.get("chatbot") or {})
    prompt_path = resolve_path(chatbot_config.get("prompt_path", str(DEFAULT_PROMPT)), PROJECT_ROOT)
    merged = {**DEFAULT_CHATBOT_CONFIG, **chatbot_config}
    return merged, prompt_path


def records_to_csv(records: list[dict]) -> str:
    if not records:
        return ""
    output = StringIO()
    fieldnames = list(records[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


with st.sidebar:
    st.header("数据库")
    db_path = Path(st.text_input("SQLite path", str(DEFAULT_DB))).expanduser()
    config_path = Path(st.text_input("Config path", str(DEFAULT_CONFIG))).expanduser()
    limit = st.number_input("结果数量", min_value=1, max_value=1000, value=100, step=10)
    top_k = st.number_input("Chatbot 检索段落数", min_value=1, max_value=20, value=8, step=1)

if not db_path.exists():
    st.warning(f"Database not found: {db_path}")
    st.code(
        "PYTHONPATH=src python -m sdg_extract.cli pipeline "
        "--config configs/sdg13_china_us_gpt56_luna_v2.yaml"
    )
    st.stop()

summary = database_summary(db_path)
st.caption(
    f"政策 {summary['policies']} 条 | 国家 {summary['countries']} 个 | "
    f"有效政策 {summary.get('valid_policies', 0)} 条 | 已审查 {summary.get('reviewed_policies', 0)} 条 | "
    f"可检索段落 {summary.get('passages', 0)} 条"
)

if config_path.exists():
    active_config = load_config(config_path)
    output_dir = resolve_path(active_config["paths"].get("output_dir", "data/output"), PROJECT_ROOT)
    source_output = output_dir / "records.jsonl"
    if source_output.exists() and database_is_stale(db_path, source_output):
        st.warning("数据库落后于最新导出结果，请运行 pipeline 或重新执行 db-build。")

browser_tab, chatbot_tab = st.tabs(["数据库浏览", "Chatbot"])

with browser_tab:
    country_options = distinct_values(db_path, "country")
    review_options = distinct_values(db_path, "review_status")
    translation_options = distinct_values(db_path, "translation_status")

    c1, c2, c3, c4 = st.columns(4)
    countries = c1.multiselect("Country", country_options)
    sdg_targets = c2.multiselect("SDG target", ["13.1", "13.2", "13.3", "13.a", "13.b"])
    review_statuses = c3.multiselect("Review status", review_options)
    translation_statuses = c4.multiselect("Translation status", translation_options)

    c5, c6 = st.columns([3, 1])
    search_text = c5.text_input("Full-text search", "")
    year = c6.text_input("Publication year", "")

    rows = query_policies(
        db_path=db_path,
        query=search_text or None,
        countries=countries,
        sdg_targets=sdg_targets,
        publication_year=year or None,
        review_statuses=review_statuses,
        translation_statuses=translation_statuses,
        limit=int(limit),
    )

    display_fields = [
        "id",
        "country",
        "policy_title_en",
        "policy_title",
        "publication_date",
        "sdg_targets",
        "review_status",
        "review_confidence",
        "translation_status",
        "source_filename",
    ]
    st.dataframe([{field: row.get(field) for field in display_fields} for row in rows], use_container_width=True)
    st.download_button(
        "Export filtered CSV",
        data=records_to_csv(rows),
        file_name="redbird_filtered_policies.csv",
        mime="text/csv",
    )

    if rows:
        selected_id = st.selectbox(
            "Policy detail",
            [str(row["id"]) for row in rows],
            format_func=lambda value: next(
                (
                    f"{row.get('country')} | {row.get('policy_title_en') or row.get('policy_title')}"
                    for row in rows
                    if str(row.get("id")) == value
                ),
                value,
            ),
        )
        selected = next(row for row in rows if str(row.get("id")) == selected_id)
        st.subheader(selected.get("policy_title_en") or selected.get("policy_title") or "Policy")
        st.write(
            {
                "country": selected.get("country"),
                "publication_date": selected.get("publication_date"),
                "sdg_targets": selected.get("sdg_targets"),
                "review_status": selected.get("review_status"),
                "review_confidence": selected.get("review_confidence"),
                "source_filename": selected.get("source_filename"),
                "source_file": selected.get("source_file"),
            }
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**Original Content**")
            st.write(selected.get("policy_content") or "")
        with right:
            st.markdown("**English Content**")
            st.write(selected.get("policy_content_en") or "")

with chatbot_tab:
    chatbot_config, prompt_path = load_chatbot_config(config_path, db_path)
    chatbot_config["top_k"] = int(top_k)
    chatbot_config["allow_general_knowledge"] = st.checkbox(
        "数据库无证据时允许使用通用知识回答",
        value=bool(chatbot_config.get("allow_general_knowledge", True)),
        help="General answers are not database evidence and do not use web search.",
    )
    chatbot_config["require_reviewed"] = st.checkbox(
        "仅使用已审查政策",
        value=bool(chatbot_config.get("require_reviewed", False)) and summary.get("reviewed_policies", 0) > 0,
        help="默认关闭。开启后，未经审查的实验记录不会参与回答。",
        disabled=summary.get("reviewed_policies", 0) == 0,
    )
    if summary.get("reviewed_policies", 0) == 0:
        st.info("当前数据库没有已审查政策，Chatbot 将检索有效但未审查的记录，并在来源中标明状态。")
    if "redbird_chat_history" not in st.session_state:
        st.session_state.redbird_chat_history = []

    clear_col, _ = st.columns([1, 5])
    if clear_col.button("Clear chat"):
        st.session_state.redbird_chat_history = []
        st.rerun()

    for message in st.session_state.redbird_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption(f"Status: {message.get('status')}")
                st.dataframe(message["sources"], use_container_width=True)

    question = st.chat_input("Ask about policies, SDG 13, or a general research topic")
    if question:
        history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.redbird_chat_history
        ]
        st.session_state.redbird_chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching policies and preparing an answer..."):
                result = answer_question(
                    db_path=db_path,
                    question=question,
                    prompt_path=prompt_path,
                    config=chatbot_config,
                    history=history,
                )
            st.markdown(result["answer"])
            st.caption(f"状态：{result['status']}")
            if result["sources"]:
                st.dataframe(result["sources"], use_container_width=True)
            with st.expander("检索详情", expanded=not bool(result["sources"])):
                st.json(result.get("retrieval", {}))
        st.session_state.redbird_chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "status": result["status"],
            }
        )
