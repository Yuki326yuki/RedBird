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
from sdg_extract.database import database_summary, distinct_values, query_policies  # noqa: E402


DEFAULT_DB = PROJECT_ROOT / "data/db/redbird_sdg13.sqlite"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/sdg13_china_us_local_100_strict_qwen3_32b_translate32b.yaml"
DEFAULT_PROMPT = PROJECT_ROOT / "prompts/chatbot/policy_qa.md"


st.set_page_config(page_title="Red Bird Policy Database", layout="wide")
st.title("Red Bird Policy Database")


def load_chatbot_config(config_path: Path, db_path: Path) -> tuple[dict, Path]:
    if not config_path.exists():
        config = dict(DEFAULT_CHATBOT_CONFIG)
        config["db_path"] = str(db_path)
        return config, DEFAULT_PROMPT
    full_config = load_config(config_path)
    chatbot_config = full_config.get("chatbot") or {}
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
    st.header("Database")
    db_path = Path(st.text_input("SQLite path", str(DEFAULT_DB))).expanduser()
    config_path = Path(st.text_input("Config path", str(DEFAULT_CONFIG))).expanduser()
    limit = st.number_input("Result limit", min_value=1, max_value=1000, value=100, step=10)
    top_k = st.number_input("Chatbot top-k", min_value=1, max_value=20, value=6, step=1)

if not db_path.exists():
    st.warning(f"Database not found: {db_path}")
    st.code(
        "PYTHONPATH=src python -m sdg_extract.cli db-build "
        "--input data/output/sdg13_china_us/local_100_strict_qwen3_32b_translate32b/records.jsonl "
        "--db data/db/redbird_sdg13.sqlite"
    )
    st.stop()

summary = database_summary(db_path)
st.caption(f"{summary['policies']} policies across {summary['countries']} countries")

browser_tab, chatbot_tab = st.tabs(["Database Browser", "Chatbot"])

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
    question = st.text_area("Ask a question about the reviewed policy database", height=120)
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Searching policies and asking the model..."):
            result = answer_question(
                db_path=db_path,
                question=question.strip(),
                prompt_path=prompt_path,
                config=chatbot_config,
            )
        st.markdown(result["answer"])
        st.caption(f"Status: {result['status']}")
        if result["sources"]:
            st.markdown("**Sources**")
            st.dataframe(result["sources"], use_container_width=True)
