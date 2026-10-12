# Red Bird：多语言 SDG 13 政策数据库与智能问答

Red Bird 用于从多国家政策文件中提取与 SDG 13 直接相关的政策措施，并依次完成小目标分类、独立审查、英文翻译、政策级聚合、SQLite 建库和 Chatbot 问答。

当前主流程为：

```text
原始文件 -> 文档解析 -> chunk 切分与候选筛选
-> 政策措施抽取 -> SDG 13 target 独立分类 -> 独立审查
-> 一份政策一行聚合 -> 英文翻译 -> CSV/JSONL -> SQLite -> Interface/Chatbot
```

## 当前配置

项目保留两份可运行样本配置：

| 模式 | 配置文件 | 数据范围 | 模型 |
|---|---|---|---|
| 本地实验 | `configs/sdg13_china_us_mexico_local_2_each_qwen25_7b_bf16.yaml` | 中国、美国、墨西哥，每国 2 份通过审查的政策 | Qwen2.5-7B-Instruct BF16 |
| API 实验 | `configs/sdg13_all_countries_gpt56_luna_v2_100.yaml` | manifest 中 20 个国家，100 个候选 chunk | gpt-5.6-luna |

本地模型路径为：

```text
/hpc2hdd/home/czhang162/models/Qwen2.5-7B-Instruct
```

旧的 1.5B、7B、32B 模型和历史结果不会被删除或覆盖，但不再作为当前默认实验。

## 安装与进入环境

首次安装：

```bash
conda activate redbird
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"
python -m pip install -e ".[pdf,yaml,dev,local,app]"
```

以后重新登录服务器时执行：

```bash
conda activate redbird
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"
export PYTHONPATH=src
```

## 一键运行本地 7B 实验

推荐使用包装脚本。它会把抽取/审查、翻译/导出、数据库构建放在独立进程中；任一步失败会停止后续步骤，最后无论成功还是失败都会启动占卡程序。

运行前应先停止已有占卡进程，然后执行：

```bash
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"
chmod +x scripts/run_local_pipeline_always_hold_gpu.sh
./scripts/run_local_pipeline_always_hold_gpu.sh
```

指定 GPU 或调整占卡参数：

```bash
GPU=1 TARGET_UTIL=60 TARGET_MEM_PERCENT=40 \
  ./scripts/run_local_pipeline_always_hold_gpu.sh
```

脚本结束后按 `Ctrl+C` 可停止占卡。运行日志位于：

```text
logs/qwen25_7b_bf16_2_each_时间.log
```

本地流程不需要 API 密钥。只有启动 Chatbot 时才需要 `OPENAI_API_KEY`。

本地样本会在三国之间轮询处理候选 chunk。每个国家得到 2 个通过审查的不同政策文件后停止处理该国，三国都达到配额后立即结束；`max_chunks: 300` 只是安全扫描上限。若 300 个候选内有国家未达到 2 份，抽取命令会明确失败并保留 checkpoint，而不会把不合格内容凑成结果。

## 分步运行本地实验

需要定位问题时，可以分三步运行。

第一步，抽取、target 分类和审查：

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n redbird env PYTHONPATH=src PYTHONUNBUFFERED=1 \
python -m sdg_extract.cli extract \
  --config configs/sdg13_china_us_mexico_local_2_each_qwen25_7b_bf16.yaml
```

第二步，政策级聚合、英文翻译和导出：

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n redbird env PYTHONPATH=src PYTHONUNBUFFERED=1 \
python -m sdg_extract.cli export \
  --config configs/sdg13_china_us_mexico_local_2_each_qwen25_7b_bf16.yaml \
  --format jsonl,csv
```

第三步，构建 SQLite：

```bash
conda run --no-capture-output -n redbird env PYTHONPATH=src \
python -m sdg_extract.cli db-build \
  --config configs/sdg13_china_us_mexico_local_2_each_qwen25_7b_bf16.yaml
```

也可以不用占卡脚本，直接执行完整 pipeline：

```bash
conda run --no-capture-output -n redbird env PYTHONPATH=src \
python -m sdg_extract.cli pipeline \
  --config configs/sdg13_china_us_mexico_local_2_each_qwen25_7b_bf16.yaml
```

## API 模式

API 样本覆盖 manifest 中的 20 个国家，并按国家、文件和 target 通道选取 100 个候选 chunk。密钥只放在环境变量中，不要写进 YAML 或 Git：

```bash
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"
export OPENAI_API_KEY="你的 API 密钥"

conda run --no-capture-output -n redbird env PYTHONPATH=src \
python -m sdg_extract.cli pipeline \
  --config configs/sdg13_all_countries_gpt56_luna_v2_100.yaml
```

API 地址和模型名由配置指定：

```text
https://api.openlux.ai/v1
gpt-5.6-luna
```

API 模式会产生抽取、分类、审查和翻译调用。开始全量任务前应先用 100-chunk 样本估算时间与费用。

## 解析与 chunk 缓存

本地 7B 配置使用两层共享缓存：

```text
data/cache/sdg13_china_us_mexico/documents.jsonl
data/cache/sdg13_china_us_mexico/documents.meta.json
data/cache/sdg13_china_us_mexico/chunks.jsonl
data/cache/sdg13_china_us_mexico/chunks.meta.json
```

当前缓存已经由旧 v3 实验中验证一致的 33 份中美墨文档迁移生成。运行时行为如下：

1. 文档指纹检查国家列表、选中的 manifest 记录、源文件路径、大小、修改时间和解析器版本。
2. 指纹一致时直接加载 `documents.jsonl`，不再解析 PDF。
3. chunk 指纹检查文档指纹、切分器版本、`max_chars=2400` 和 `overlap_chars=300`。
4. 参数一致时直接加载 `chunks.jsonl`；只有切分参数或源文件变化时才重建。
5. 缓存不包含模型输出，可以安全地被不同本地实验复用。

日志中出现下面两行即表示解析和切分均已复用：

```text
[CACHE] documents hit; PDF parsing skipped.
[CACHE] chunks hit; document splitting skipped.
```

## 推理速度与显存

本地后端直接调用 `model.generate()`，不再使用通用 Transformers pipeline。配置使用：

```yaml
torch_dtype: auto
attn_implementation: sdpa
quantization:
  mode: none
```

Qwen2.5-7B 原模型为 BF16，单卡加载显存通常应低于 20 GiB。四个阶段分别限制最大输出：

```text
抽取 512 tokens
target 分类 700 tokens
审查 350 tokens
翻译 1800 tokens
```

抽取、分类和审查在同一进程复用同一份 7B 权重，不会加载三次。翻译在下一个独立进程重新加载一次，确保上一阶段显存已释放。

每次模型调用都会输出：

```text
[MODEL] stage=... input_tokens=... output_tokens=... elapsed_sec=... tokens_per_sec=...
[GPU] stage=... allocated_gib=... reserved_gib=...
```

前三个待处理 chunk 是速度预检。任一 chunk 的完整三阶段处理超过 180 秒会停止实验并提示 `Speed preflight failed`，避免长时间空跑。输入超过 6144 tokens 会明确报错，不会静默截断证据。

## 断点续跑

抽取进度保存在新实验 workdir 的 `checkpoint.json`：

- `completed_chunk_ids` 只包含成功完成的 chunk，再次运行会跳过。
- `failed_chunk_ids` 不会当作完成，下一次运行会重试。
- `records.jsonl` 每完成一个 chunk 就更新。
- target 分类和审查分别使用带模型名与 prompt 哈希的缓存。
- 翻译每完成一份政策就原子更新 `translations.jsonl`，中断后可以继续。

恢复运行时直接重复原命令即可，不需要重新解析 PDF，也不要删除共享缓存。

如果需要从零开始做一个不同模型或 prompt 的实验，请使用新的 `workdir`、`output_dir` 和 `db_path`，不要把旧 checkpoint 与新逻辑混用。

## SDG 13 提取与分类逻辑

1. 解析器按页保存段落、页码和文本质量信息。
2. 多语言关键词只对正文进行候选筛选，标题和文件名不作为气候相关证据。
3. 第一阶段最多抽取两条明确政策措施，不判断 target；正文在 prompt 中只出现一次。
4. 第二阶段分别判断 `13.1`、`13.2`、`13.3`、`13.a`、`13.b`，每个目标可为 `supported`、`insufficient` 或 `contradicted`。
5. 同一政策措施可以支持多个 target，也可以一个都不支持。
6. 确定性规则要求证据原文可追溯、target 合法、置信度至少为 0.75。
7. 第三阶段独立审查并输出 `accept`、`revise` 或 `reject`；失败和低置信度记录不进入正式结果。

`13.2` 不能作为默认分类。它必须有将气候措施纳入政策、战略、规划、法律、监管、NDC 或国家实施机制的明确证据。一般排放、能源、绿色发展、科技或经济内容不足以判为 `13.2`。

## 输出位置

本地 7B 最终结果：

```text
data/output/sdg13_china_us_mexico/local_2_each_qwen25_7b_bf16/records.csv
data/output/sdg13_china_us_mexico/local_2_each_qwen25_7b_bf16/records.jsonl
data/db/redbird_sdg13_china_us_mexico_2_each_qwen25_7b_bf16.sqlite
```

API 样本结果：

```text
data/output/sdg13_all_countries/gpt56_luna_v2_100/records.csv
data/output/sdg13_all_countries/gpt56_luna_v2_100/records.jsonl
data/db/redbird_sdg13_all_countries_v2.sqlite
```

最终 CSV/JSONL 是 policy-level 数据，一份政策文件一行，字段包括：

```text
country, country_iso3, policy_title, policy_title_en, publication_date,
policy_content, policy_content_en, policy_summary, policy_summary_en,
sdg_goal, sdg_targets, source_filename, source_file, record_count,
review_status, review_confidence, translation_model,
translation_status, validation_status
```

原文字段保留政策原语言，`*_en` 字段为英文。逐 chunk 的证据和页码留在 workdir 及数据库 passages 表中，不写入政策主表。

## 质量与调试文件

本地实验工作目录：

```text
data/work/sdg13_china_us_mexico/local_2_each_qwen25_7b_bf16/
```

其中包括：

- `records.jsonl`：通过分类和审查的 chunk-level 记录。
- `checkpoint.json`：成功和待重试 chunk。
- `target_classifications.jsonl`：五个 target 的分类缓存。
- `reviews.jsonl`：审查决定及拒绝原因。
- `translations.jsonl`：政策级英文翻译缓存。
- `errors.jsonl`：模型、JSON、OOM 或审查错误。
- `quality_report.json`：target 数量、占比、国家分布和集中度警告。

查看 target 分布：

```bash
cat data/work/sdg13_china_us_mexico/local_2_each_qwen25_7b_bf16/quality_report.json
```

集中度警告用于提醒人工抽查，不会为了数量均衡而修改模型判断。

## 数据库与命令行检索

SQLite 包含：

- `policies`：一份政策一行。
- `passages`：相关政策段落、页码和审查状态。
- `policy_targets`：政策与 target 的多对多关系。
- `policies_fts` 和 `passages_fts`：中英文全文索引。
- `db_meta`：源输出哈希、构建时间和 schema 版本。

命令行检索示例：

```bash
conda run -n redbird env PYTHONPATH=src \
python -m sdg_extract.cli db-query \
  --db data/db/redbird_sdg13_china_us_mexico_2_each_qwen25_7b_bf16.sqlite \
  --q "climate adaptation" --country China --limit 8
```

本项目不使用 BGE-M3 或向量数据库。检索使用国家/年份/target 的 SQL 过滤、多语言别名、原文和英文 FTS5、查询扩展、翻译桥接及 API 重排。

## 启动 Interface 与 Chatbot

先设置 Chatbot API 密钥：

```bash
export OPENAI_API_KEY="你的 API 密钥"
```

启动界面：

```bash
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"
conda run --no-capture-output -n redbird env PYTHONPATH=src \
streamlit run app/redbird_interface.py
```

浏览器访问：

```text
http://localhost:8501
```

在侧栏填写需要使用的 SQLite 路径。界面支持国家、target、年份、审查状态和翻译状态筛选，查看中英文详情，并导出当前结果。

Chatbot 有五种回答模式：

- `general`：直接回答通用问题。
- `database_search`：根据命中的政策 passage 回答并引用来源。
- `database_aggregate`：总结全部政策或统计数量。
- `compare`：按国家、年份或 target 分组比较。
- `mixed`：把数据库事实与通用解释分开说明。

数据库无命中时，系统只说明本次检索没有找到证据，不会断言政策不存在；开启通用知识后仍会继续回答。界面的“检索详情”会展示识别意图、过滤条件、查询变体、各阶段命中数和最终 passage。

## 运行测试

```bash
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"
conda run -n redbird env PYTHONPATH=src \
python -m unittest discover -s tests
```

## 常见问题

### 显卡占用为 0

先确认当前运行的是本地配置。API 配置和 Chatbot 调用远程模型，本地 GPU 占用为 0 是正常现象。本地 7B 启动时会先加载约 15 GB 权重，之后日志应持续出现 `[MODEL]` 和 `[GPU]`。

### 命令立即结束

通常表示该 workdir 的 checkpoint 已经完成。查看 `checkpoint.json` 和日志；继续同一实验可直接复用结果，需要重新实验则换一个新 workdir。

### 出现 OOM

先停止其他训练或占卡进程，再确认 `CUDA_VISIBLE_DEVICES` 指向空闲 GPU。当前 7B BF16 配置已经使用 SDPA、较短阶段输出和 OOM 降档重试，不应像 32B 那样占用大量显存。

### Chatbot 一直说没有数据库证据

检查：

- 侧栏数据库路径是否指向最新 SQLite。
- `policies` 和 `passages` 是否大于 0。
- 是否误开“仅使用已审查政策”。
- 是否在导出后重新执行了 `db-build`。
- “检索详情”中的结构化条件和 FTS 命中数是否正常。

### API 调用失败

检查 `OPENAI_API_KEY`、OpenLux 余额、网络、`api_base` 和模型名是否可用。Responses API 不兼容时程序会回退到 Chat Completions，但认证、余额和限流错误会明确显示。

## 扩展到全量

先完成每国 2 份政策的本地实验或 100-chunk API 实验，并人工检查相关性、target 分布、审查拒绝率和翻译质量。确认后复制配置到新文件，删除政策配额或 `experiment.max_chunks`，并设置新的 `workdir`、`output_dir` 和 `db_path`。不要覆盖当前样本结果，也不要复用其模型 checkpoint。
