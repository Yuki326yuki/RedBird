# Red Bird：SDG 政策提取、审查、翻译与数据库问答工具

Red Bird 是一个面向政策文本的本地研究工具链。它可以从本地 PDF、HTML、TXT、DOCX、CSV 等政策文件中提取与 SDG 相关的政策内容，并进一步完成严格审查、政策级聚合、英文翻译、数据库构建、界面浏览和 Chatbot 问答。

当前项目主要围绕 SDG 13 气候行动实验配置展开，支持中国和美国政策数据的 100 chunk 小规模实验，也可以扩展到更大的政策集合。

## 主要功能

- 文档摄入：读取本地政策文件，解析为标准化 `documents.jsonl` 和 `chunks.jsonl`。
- chunk 级抽取：将长政策文件切分为 chunk，再调用模型抽取 SDG 相关政策记录。
- SDG13 严格过滤：仅保留明确涉及气候行动的内容。
- 双阶段审查：抽取后使用独立审查 prompt 判断记录是否真正支持 SDG 13 与具体 target。
- policy-level 聚合：最终导出为“一份政策文件一行”，更适合数据库使用。
- 英文翻译：将政策标题、政策内容、政策摘要统一翻译为英文。
- SQLite 数据库：将最终 `records.jsonl/csv` 导入 SQLite，并建立 FTS5 全文检索索引。
- 本地界面：使用 Streamlit 浏览、筛选、搜索和导出政策表。
- Chatbot：基于数据库检索结果回答用户问题，并显示引用来源。

## 环境准备

推荐使用已经创建好的 conda 环境：

```bash
conda activate redbird
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"
```

如果没有激活环境，也可以使用 `conda run`：

```bash
conda run -n redbird env PYTHONPATH=src python -m sdg_extract.cli --help
```

项目本地运行时通常需要设置：

```bash
export PYTHONPATH=src
```

如果要使用 API 模型或 Chatbot，需要设置 API key：

```bash
export OPENAI_API_KEY="你的 API key"
```

## 配置文件说明

常用配置位于 `configs/`：

- `configs/default.yaml`：默认 API 模式配置。
- `configs/local.yaml`：本地模型配置。
- `configs/sdg13_china_us_local.yaml`：中美 SDG13 本地模型实验配置。
- `configs/sdg13_china_us_local_100.yaml`：中美 100 chunk 测试配置。
- `configs/sdg13_china_us_local_100_strict.yaml`：中美 100 chunk 严格 SDG13 配置。
- `configs/sdg13_china_us_local_100_strict_qwen25_7b.yaml`：使用 Qwen2.5-7B 的严格测试配置。
- `configs/sdg13_china_us_local_100_strict_qwen3_32b_translate32b.yaml`：使用 Qwen3-32B 抽取、审查和翻译的完整实验配置。

当前推荐的完整流程配置是：

```text
configs/sdg13_china_us_local_100_strict_qwen3_32b_translate32b.yaml
```

该配置包含：

- 输入目录：`data/sdg13_policy_collection/original_documents`
- 工作目录：`data/work/sdg13_china_us/local_100_strict_qwen3_32b_translate32b`
- 输出目录：`data/output/sdg13_china_us/local_100_strict_qwen3_32b_translate32b`
- 抽取模型：`/hpc2hdd/home/czhang162/models/Qwen3-32B`
- 审查模型：`/hpc2hdd/home/czhang162/models/Qwen3-32B`
- 翻译模型：`/hpc2hdd/home/czhang162/models/Qwen3-32B`
- Chatbot 模型：OpenAI-compatible API，默认 `gpt-4.1-mini`

## 基础命令

查看所有 CLI 命令：

```bash
PYTHONPATH=src python -m sdg_extract.cli --help
```

当前支持：

- `ingest`：解析本地文档。
- `extract`：执行抽取、过滤和审查。
- `export`：导出 policy-level JSONL/CSV，并可执行翻译。
- `db-build`：从最终导出结果构建 SQLite 数据库。
- `db-query`：在命令行中测试数据库检索。

## 运行完整 32B 严格实验

进入项目目录：

```bash
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"
```

执行抽取和审查：

```bash
conda run --no-capture-output -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli extract \
  --config configs/sdg13_china_us_local_100_strict_qwen3_32b_translate32b.yaml
```

执行导出和英文翻译：

```bash
conda run --no-capture-output -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli export \
  --config configs/sdg13_china_us_local_100_strict_qwen3_32b_translate32b.yaml \
  --format jsonl,csv
```

最终输出位于：

```text
data/output/sdg13_china_us/local_100_strict_qwen3_32b_translate32b/records.jsonl
data/output/sdg13_china_us/local_100_strict_qwen3_32b_translate32b/records.csv
```

中间文件位于：

```text
data/work/sdg13_china_us/local_100_strict_qwen3_32b_translate32b/
```

重要中间文件：

- `documents.jsonl`：解析后的文档。
- `chunks.jsonl`：切分后的 chunk。
- `records.jsonl`：通过审查后的 chunk-level 抽取记录。
- `reviews.jsonl`：审查日志，包括 accept、revise、reject。
- `translations.jsonl`：翻译缓存。
- `checkpoint.json`：抽取进度。
- `errors.jsonl`：抽取或审查错误日志。

## 输出字段说明

最终 `records.csv/jsonl` 是 policy-level 结果，即一份政策文件一行。

字段包括：

- `country`：国家。
- `country_iso3`：国家 ISO3 代码。
- `policy_title`：政策标题原文。
- `policy_title_en`：政策标题英文。
- `publication_date`：发布日期。
- `policy_content`：与 SDG 相关的政策内容原文。
- `policy_content_en`：政策内容英文。
- `policy_summary`：政策摘要原文。
- `policy_summary_en`：政策摘要英文。
- `sdg_goal`：SDG 目标，例如 `SDG 13`。
- `sdg_targets`：相关 SDG target，例如 `13.1; 13.2`。
- `source_filename`：来源文件名。
- `source_file`：来源文件完整路径。
- `record_count`：该政策由多少条 chunk-level 记录聚合而来。
- `review_status`：审查状态。
- `review_confidence`：审查平均置信度。
- `translation_model`：翻译模型。
- `translation_status`：翻译状态。
- `validation_status`：字段校验状态。

## 抽取逻辑说明

项目不会把整份 PDF 一次性直接交给模型，而是先解析并切分为 chunk。原因是政策 PDF 通常很长，超过模型上下文限制；同时 chunk 能保留更清晰的证据位置，便于审查和追溯。

当前流程是：

```text
PDF/HTML/TXT/DOCX/CSV
  -> parse documents
  -> split chunks
  -> SDG13 keyword candidate filter
  -> model extraction
  -> strict SDG13 filter
  -> independent review
  -> chunk-level records
  -> policy-level aggregation
  -> English translation
  -> records.csv/jsonl
```

最终表不是“每个 chunk 一行”，而是“一份政策文件一行”。chunk 只作为内部抽取和证据定位单位。

## 审查机制说明

审查阶段使用独立 prompt：

```text
prompts/review/sdg13_strict.md
```

审查输入包括：

- 原始 chunk 文本。
- 候选抽取记录。
- 允许的 SDG13 target 定义。

审查输出包括：

- `accept`：记录完全由原文支持。
- `revise`：记录相关，但需要修正 target、证据 quote 或政策内容。
- `reject`：记录证据不足、弱相关或 target 不成立。

被拒绝的记录不会进入最终数据库主表，但会保存在：

```text
data/work/.../reviews.jsonl
```

## 翻译机制说明

翻译 prompt 位于：

```text
prompts/translation/en_policy.md
```

翻译字段：

- `policy_title -> policy_title_en`
- `policy_content -> policy_content_en`
- `policy_summary -> policy_summary_en`

翻译要求：

- 忠实翻译，不扩写。
- 保留数字、年份、百分比、机构名、NDC、UNFCCC、Paris Agreement 等术语。
- 已经是英文的内容会返回清洁英文版本。
- 空字段返回空。

翻译缓存保存在：

```text
data/work/.../translations.jsonl
```

重复导出时会优先读取缓存，避免重复调用模型。

## 构建 SQLite 数据库

从最终导出的 `records.jsonl` 构建数据库：

```bash
conda run -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli db-build \
  --input data/output/sdg13_china_us/local_100_strict_qwen3_32b_translate32b/records.jsonl \
  --db data/db/redbird_sdg13.sqlite
```

如果当前还没有 32B 输出，也可以先用已有 7B 输出测试：

```bash
conda run -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli db-build \
  --input data/output/sdg13_china_us/local_100_strict_qwen25_7b/records.jsonl \
  --db data/db/redbird_sdg13.sqlite
```

数据库包含：

- `policies`：政策主表。
- `policies_fts`：全文检索索引。

数据库默认路径：

```text
data/db/redbird_sdg13.sqlite
```

## 命令行检索数据库

示例：

```bash
conda run -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli db-query \
  --db data/db/redbird_sdg13.sqlite \
  --q "methane emissions" \
  --limit 5
```

按国家过滤：

```bash
conda run -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli db-query \
  --db data/db/redbird_sdg13.sqlite \
  --q "emissions" \
  --country "United States"
```

按 SDG target 过滤：

```bash
conda run -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli db-query \
  --db data/db/redbird_sdg13.sqlite \
  --q "adaptation" \
  --sdg-target 13.1
```

## 启动本地界面

启动 Streamlit：

```bash
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"
conda run -n redbird streamlit run app/redbird_interface.py
```

界面功能：

- 查看数据库总量。
- 按国家筛选。
- 按 SDG target 筛选。
- 按年份筛选。
- 按 `review_status` 筛选。
- 按 `translation_status` 筛选。
- 全文检索政策标题、内容、摘要。
- 查看单条政策详情。
- 查看原文和英文内容。
- 导出当前筛选结果为 CSV。
- 使用 Chatbot 基于数据库问答。

如果界面提示数据库不存在，请先运行 `db-build`。

## 使用 Chatbot

Chatbot 位于 Streamlit 界面的 `Chatbot` 标签页。

使用前需要设置 API key：

```bash
export OPENAI_API_KEY="你的 API key"
```

Chatbot 配置在：

```yaml
chatbot:
  enabled: true
  db_path: data/db/redbird_sdg13.sqlite
  prompt_path: prompts/chatbot/policy_qa.md
  top_k: 6
  model:
    backend: api
    name: gpt-4.1-mini
    api_base: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    temperature: 0
    max_tokens: 1500
    json_mode: false
```

Chatbot 的工作方式：

```text
用户问题
  -> SQLite FTS 检索 top-k 政策记录
  -> 将检索结果作为上下文
  -> 调用 API 模型回答
  -> 显示答案和来源政策
```

Chatbot 不会直接读取 PDF，也不会使用数据库外部信息。若数据库中没有足够证据，它会提示证据不足。

## 常用实验命令

运行 7B strict 100 实验：

```bash
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"

conda run --no-capture-output -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli extract \
  --config configs/sdg13_china_us_local_100_strict_qwen25_7b.yaml

conda run --no-capture-output -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli export \
  --config configs/sdg13_china_us_local_100_strict_qwen25_7b.yaml \
  --format jsonl,csv

python3 /hpc2hdd/home/czhang162/MyProject/zhanka/占卡.py
```

运行 32B strict + review + translation 实验：

```bash
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"

conda run --no-capture-output -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli extract \
  --config configs/sdg13_china_us_local_100_strict_qwen3_32b_translate32b.yaml

conda run --no-capture-output -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli export \
  --config configs/sdg13_china_us_local_100_strict_qwen3_32b_translate32b.yaml \
  --format jsonl,csv

python3 /hpc2hdd/home/czhang162/MyProject/zhanka/占卡.py
```

构建数据库并启动界面：

```bash
conda run -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli db-build \
  --input data/output/sdg13_china_us/local_100_strict_qwen3_32b_translate32b/records.jsonl \
  --db data/db/redbird_sdg13.sqlite

conda run -n redbird streamlit run app/redbird_interface.py
```

## 测试

运行全部单元测试：

```bash
conda run -n redbird env PYTHONPATH=src \
  python -m unittest discover -s tests
```

当前测试覆盖：

- 文档解析。
- manifest 过滤。
- chunk 过滤。
- strict SDG13 过滤。
- JSON 解析容错。
- 审查 accept/revise/reject。
- policy-level 聚合。
- 翻译缓存和失败处理。
- SQLite 导入和全文检索。
- Chatbot 检索上下文和 API 失败处理。

## 常见问题

### 为什么不用整份 PDF 直接问模型？

本地模型上下文有限，很多政策 PDF 很长，直接输入整份文件容易超上下文、漏内容或产生幻觉。chunk 化可以降低输入长度，并让每条结果能追溯到具体文本片段。

### 最终结果是一份政策一行，还是一个 chunk 一行？

最终 `data/output/.../records.csv/jsonl` 是一份政策一行。chunk-level 结果只保存在 `data/work/.../records.jsonl` 中，用于审计。

### evidence_quote 为什么不在最终表里？

最终表是数据库主表，保留核心字段。逐条证据 quote 会让表过于杂乱。审计证据可以回看 workdir 中的 chunk-level `records.jsonl` 和 `reviews.jsonl`。

### 32B 很慢怎么办？

先用 100 chunk 配置测试。如果显存或速度不够，可以使用 7B 配置：

```text
configs/sdg13_china_us_local_100_strict_qwen25_7b.yaml
```

### Chatbot 查不到结果怎么办？

先确认数据库已构建：

```bash
ls -lah data/db/redbird_sdg13.sqlite
```

再用命令行检索测试：

```bash
conda run -n redbird env PYTHONPATH=src \
  python -m sdg_extract.cli db-query \
  --db data/db/redbird_sdg13.sqlite \
  --q "emissions"
```

如果命令行能查到，界面和 Chatbot 通常也能查到。

### Chatbot API 报错怎么办？

检查 `OPENAI_API_KEY`：

```bash
echo $OPENAI_API_KEY
```

如果使用其他 OpenAI-compatible 服务，需要修改配置中的：

```yaml
api_base
api_key_env
name
```
