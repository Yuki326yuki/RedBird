# Red Bird：多语言 SDG 13 政策数据库与智能问答

Red Bird 从多国家政策文件中提取与 SDG 13 直接相关的政策措施，完成 target 分类、独立审查、英文翻译、政策级聚合、SQLite 检索和 Chatbot 问答。项目支持保留原文，同时把结果统一翻译为英文，最终形成一份政策一行的数据库主表。

## 数据范围

`data/sdg13_policy_collection/download_manifest.csv` 当前包含 155 条记录、20 个国家：Canada、Mexico、China、Brazil、Vietnam、United Kingdom、United States、Egypt、Germany、Indonesia、Nepal、India、Japan、Mozambique、Philippines、Uganda、Kenya、Madagascar、South Africa 和 Ethiopia。

manifest 中已标注的语言代码包括：

```text
eng, spa, por, vie, fra, zho, cmn, deu, ara, ind, npi, jpn
```

另有少量记录未填写语言。程序不会把未标注语言强行推断为某种语言；模型仍会阅读原文并在翻译阶段生成英文列。

## 主要功能

- 解析 PDF、HTML、TXT、DOCX 和 CSV，记录页码、段落及解析质量。
- 将长文档切为带重叠的 chunk，超长段落继续按句子拆分。
- 使用多语言 SDG 13 关键词进行低成本候选召回，不使用文件名或标题冒充正文证据。
- 将政策措施抽取、五个 target 分类和最终审查拆成独立步骤。
- 支持同一政策对应多个 target，不再强制选择一个 target。
- 将通过审查的 chunk 记录聚合成一份政策一行，并统一翻译为英文。
- 建立 `policies + passages + policy_targets` SQLite 数据库和 FTS5 索引。
- 使用多语言别名、结构化 SQL、原文/英文 FTS、翻译桥接、查询扩展和 Luna 重排进行检索。
- 提供 Streamlit 数据浏览器和具备数据库知识的通用 Chatbot。

## 快速开始

下面提供三种运行方式。第一次使用建议先运行 100-chunk 实验，确认结果后再扩展到全量。

### 第一步：进入环境

```bash
conda activate redbird
cd "/hpc2hdd/home/czhang162/MyProject/Red Bird"
python -m pip install -e ".[pdf,yaml,dev,local,app]"
export PYTHONPATH=src
```

安装只需要执行一次。以后重新登录服务器时，只需要重新执行 `conda activate`、`cd` 和 `export PYTHONPATH=src`。

### 第二步：选择运行模式

可选择以下任意一种模式：

- 本地 32B：抽取、分类、审查和翻译都使用本地 `Qwen3-32B`，不消耗 API。
- API 中美实验：使用 `gpt-5.6-luna` 处理中国和美国数据。
- API 多国家实验：使用 `gpt-5.6-luna` 对 20 个国家均衡抽取 100 个候选 chunk。

不要同时运行多个模式。每种模式使用独立目录，不会覆盖其他实验结果。

### 模式 A：本地 Qwen3-32B

适合在有足够显存的 GPU 服务器上运行。完整流程只需要执行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python -m sdg_extract.cli pipeline \
  --config configs/sdg13_china_us_local_100_strict_qwen3_32b_v2.yaml
```

该命令会依次完成：

```text
PDF 解析 -> chunk 筛选 -> 政策证据抽取 -> 五个 target 独立分类
-> 独立审查 -> 政策级聚合 -> 英文翻译 -> CSV/JSONL -> SQLite
```

本地模式不需要 `OPENAI_API_KEY`。只有之后使用 Chatbot 时才需要 API 密钥。

本地模式结果位置：

```text
data/output/sdg13_china_us/local_100_strict_qwen3_32b_v2/records.csv
data/output/sdg13_china_us/local_100_strict_qwen3_32b_v2/records.jsonl
data/db/redbird_sdg13_qwen3_32b_v2.sqlite
```

质量和日志位置：

```text
data/work/sdg13_china_us/local_100_strict_qwen3_32b_v2/quality_report.json
data/work/sdg13_china_us/local_100_strict_qwen3_32b_v2/target_classifications.jsonl
data/work/sdg13_china_us/local_100_strict_qwen3_32b_v2/reviews.jsonl
data/work/sdg13_china_us/local_100_strict_qwen3_32b_v2/errors.jsonl
```

### 模式 B：API 中美实验

API 模式的抽取、分类、审查和翻译都调用 OpenLux，不占用本地 GPU。

先在当前终端设置密钥：

```bash
export OPENAI_API_KEY="你的 API 密钥"
```

不要把真实密钥写入 YAML、README 或提交到 Git。然后运行：

```bash
PYTHONPATH=src python -m sdg_extract.cli pipeline \
  --config configs/sdg13_china_us_gpt56_luna_v2.yaml
```

API 配置为：

```text
模型：gpt-5.6-luna
地址：https://api.openlux.ai/v1
密钥变量：OPENAI_API_KEY
```

中美 API 结果位置：

```text
data/output/sdg13_china_us/gpt56_luna_v2_100/records.csv
data/output/sdg13_china_us/gpt56_luna_v2_100/records.jsonl
data/db/redbird_sdg13_v2.sqlite
```

### 模式 C：API 多国家实验

该模式读取 manifest 中全部 20 个国家，并按国家、文件和 target 通道均衡选择 100 个候选 chunk：

```bash
export OPENAI_API_KEY="你的 API 密钥"

PYTHONPATH=src python -m sdg_extract.cli pipeline \
  --config configs/sdg13_all_countries_gpt56_luna_v2_100.yaml
```

多国家 API 结果位置：

```text
data/output/sdg13_all_countries/gpt56_luna_v2_100/records.csv
data/output/sdg13_all_countries/gpt56_luna_v2_100/records.jsonl
data/db/redbird_sdg13_all_countries_v2.sqlite
```

### 分步运行

通常直接运行 `pipeline` 即可。如果需要定位错误，可以分成三步。下面以本地 32B 为例。

第一步，抽取、target 分类和审查：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python -m sdg_extract.cli extract \
  --config configs/sdg13_china_us_local_100_strict_qwen3_32b_v2.yaml
```

第二步，政策级聚合、英文翻译和导出：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python -m sdg_extract.cli export \
  --config configs/sdg13_china_us_local_100_strict_qwen3_32b_v2.yaml \
  --format jsonl,csv
```

第三步，构建 SQLite 数据库：

```bash
PYTHONPATH=src python -m sdg_extract.cli db-build \
  --input data/output/sdg13_china_us/local_100_strict_qwen3_32b_v2/records.jsonl \
  --passages data/work/sdg13_china_us/local_100_strict_qwen3_32b_v2/records.jsonl \
  --db data/db/redbird_sdg13_qwen3_32b_v2.sqlite \
  --config configs/sdg13_china_us_local_100_strict_qwen3_32b_v2.yaml
```

API 模式也可以分步运行，只需把三条命令中的 `--config` 换成对应 API 配置，并提前设置 `OPENAI_API_KEY`。

### 启动数据库界面和 Chatbot

无论政策数据由本地模型还是 API 生成，Interface 的启动方式相同。Chatbot 使用 OpenLux API，因此需要先设置密钥：

```bash
export OPENAI_API_KEY="你的 API 密钥"

conda run --no-capture-output -n redbird env PYTHONPATH=src \
streamlit run app/redbird_interface.py
```

浏览器访问：

```text
http://localhost:8501
```

在界面左侧填写需要使用的 SQLite 文件：

```text
本地 32B：data/db/redbird_sdg13_qwen3_32b_v2.sqlite
API 中美：data/db/redbird_sdg13_v2.sqlite
API 多国家：data/db/redbird_sdg13_all_countries_v2.sqlite
```

Interface 可以浏览、筛选和导出数据库。Chatbot 会先检索数据库，再结合检索结果回答；普通知识问题可以直接使用模型回答。

### 查看 target 分布

每次抽取结束后都会生成 `quality_report.json`。例如：

```bash
cat data/work/sdg13_china_us/local_100_strict_qwen3_32b_v2/quality_report.json
```

重点检查：

```text
target_counts：各 target 的记录数量
target_shares：各 target 的占比
dominant_target：占比最高的 target
warnings：是否出现 target 过度集中的提醒
```

集中度警告只表示需要人工抽查，不会为了让数量均衡而修改模型结果。

### 运行测试

```bash
conda run -n redbird env PYTHONPATH=src \
python -m unittest discover -s tests
```

## 抽取逻辑

1. 解析器按页读取文件并记录文本质量；低文本量、乱码或解析异常进入 `parse_warnings.jsonl`。
2. 文档按约 3200 字符切块，相邻块重叠约 400 字符，避免政策动作与依据落在不同块。
3. 候选筛选只检查正文，并覆盖 manifest 中的英语、西班牙语、葡萄牙语、越南语、法语、中文、德语、阿拉伯语、印尼语、尼泊尔语和日语关键词。
4. 第一次 Luna 调用只提取明确、可逐字追溯的政策措施，不判断 target。
5. 第二次 Luna 调用分别检查 `13.1 / 13.2 / 13.3 / 13.a / 13.b`，每个目标返回 supported、insufficient 或 contradicted。
6. 确定性规则检查 evidence 是否确实存在于原 chunk，target 是否有对应信号，confidence 是否达到 0.75。
7. 第三次 Luna 调用独立审查，可 accept、revise 或 reject。低置信度和无法追溯的记录不进入正式结果。

13.2 已特别收紧：必须出现气候政策、气候战略、气候规划、气候法律、NDC、将气候措施纳入主流等明确整合信号。只有一般排放、能源、绿色发展、科技或经济内容不能判为 13.2。

## 工作目录

每个实验的 workdir 包含：

- `documents.jsonl`：标准化文档及解析元数据。
- `chunks.jsonl`：带页码、段落 ID 和语言信息的 chunk。
- `parse_warnings.jsonl`：解析质量警告。
- `records.jsonl`：通过分类和审查的 chunk-level 记录。
- `target_classifications.jsonl`：五目标分类缓存。
- `reviews.jsonl`：accept、revise、reject 审查日志。
- `translations.jsonl`：政策级翻译缓存。
- `checkpoint.json`：抽取进度。
- `errors.jsonl`：模型或解析错误。
- `quality_report.json`：target 数量、占比、国家分布和集中度警告。

修改 prompt、模型或规则时应使用新的 workdir，避免旧 checkpoint 和新逻辑混用。

## 最终输出

`data/output/.../records.jsonl` 和 `records.csv` 均为 policy-level 结果，一份政策文件一行。主要字段包括：

```text
country, country_iso3, policy_title, policy_title_en, publication_date,
policy_content, policy_content_en, policy_summary, policy_summary_en,
sdg_goal, sdg_targets, source_filename, source_file, record_count,
review_status, review_confidence, translation_model,
translation_status, validation_status
```

原文字段保留政策原语言，`*_en` 字段统一为英文。逐 chunk 的证据、页码和审查记录保留在 workdir 与数据库 `passages` 表，不塞入最终主表。

## 数据库结构

- `policies`：一份政策一行。
- `passages`：已抽取的相关段落、原文语言、页码和审查状态。
- `policy_targets`：政策与 SDG target 的多对多关系及证据数量。
- `policies_fts`：索引政策标题、摘要、原文和英文翻译。
- `passages_fts`：索引实际命中的政策段落。
- `db_meta`：源文件哈希、构建时间、schema 版本和检索后端。

数据库先构建临时文件，全部成功后原子替换正式 SQLite。界面会检查输出哈希，数据库落后时显示警告。

## 无向量模型的多语言检索

本项目不使用 BGE-M3，也不依赖任何本地 embedding 模型。检索按以下顺序完成：

1. 确定性规划器识别国家别名、年份、SDG goal/target、总结、数量和比较意图。
2. Luna 查询规划器将问题改写为原语言查询和英文查询。
3. 国家、年份和 target 使用 SQL 精确过滤，不混入全文关键词。
4. FTS5 同时检索政策原文、英文翻译和 passage。
5. 如果英文翻译命中 policy 主表，系统通过 policy ID 找回对应的原文 passage，这就是跨语言“翻译桥接”。
6. 常见气候术语在 12 种语言间扩展；FTS 先尝试 AND，再降级为 OR。
7. 最多取 20 个候选 passage，由 Luna 依据用户问题重排，最终送入回答模型的默认数量为 8。
8. “全部、总结、数量、比较”类问题直接走 SQL 聚合，不受 top-k 限制。

这种方案不需要下载额外模型，也不会改动现有本地模型目录。

## 启动界面

```bash
conda run -n redbird env PYTHONPATH=src \
  streamlit run app/redbird_interface.py
```

通常访问：

```text
http://localhost:8501
```

界面支持国家、target、年份、审查状态和翻译状态筛选，查看中英文详情，导出当前结果，以及多轮 Chatbot。展开“检索详情”可以看到识别意图、结构化条件、查询变体、FTS 命中数、翻译桥接命中数、最终 passage 数和降级阶段。

## Chatbot 回答模式

- `general`：普通知识问题，不依赖数据库。
- `database_search`：检索具体政策 passage，并列出政策、国家、target、页码、文件名和审查状态。
- `database_aggregate`：总结全部政策或统计数量，使用完整 SQL 结果集合。
- `compare`：按国家、年份或 target 分组比较。
- `mixed`：区分数据库事实与模型通用解释。

数据库无命中时，系统只会说“本次数据库检索没有找到证据”，不会断言政策不存在；开启通用知识后仍会继续作答。默认允许 `validation_status=valid` 但尚未审查的实验记录参与检索，并明确标记为 `unreviewed`。只有用户主动勾选“仅使用已审查政策”时才严格过滤。

## 扩展到全量数据

先用 20 国 100-chunk 配置检查召回、target 分布、审查拒绝率和翻译质量。确认后复制该配置到新文件，删除 `experiment.max_chunks: 100`，并使用新的 workdir、output_dir 和 db_path。不要直接复用小实验目录。

全量任务会产生多次 Luna API 调用：每个候选至少经历抽取、target 分类和审查，导出时还会翻译。建议先估算 API 成本和运行时间。

## 测试

```bash
conda run -n redbird env PYTHONPATH=src \
  python -m unittest discover -s tests
```

测试覆盖解析质量、重叠分块、多语言候选过滤、13.2 去偏、证据追溯、审查、翻译、政策级聚合、passage 数据库、跨语言翻译桥接、未审查记录召回、SQL 汇总和 Chatbot 错误处理。

## 常见问题

### Chatbot 总说数据库没有证据

检查界面的“检索详情”：

- 数据库路径是否正确，`policies` 和 `passages` 是否大于 0。
- 是否误开“仅使用已审查政策”。
- 数据库是否落后于最新 `records.jsonl`。
- 查询是否识别出正确国家、target 和模式。
- `lexical_hits`、`policy_bridge_hits` 与最终命中数分别是多少。

重新导出后，应重新运行 pipeline 或 `db-build` 同步数据库。

### API 返回空内容

检查 `OPENAI_API_KEY`、OpenLux 余额、`api_base` 和 `gpt-5.6-luna` 是否可用。Responses API 不兼容时会回退 Chat Completions，但认证和限流错误不会被掩盖。

### 重新运行立刻结束

对应 workdir 的 `checkpoint.json` 已记录处理完成。需要重新实验时请改用新目录，不要删除或覆盖其他实验结果。

### 显卡占用为 0

GPT-5.6 Luna 是远程 API 模型，本地 GPU 不参与抽取、审查、翻译或 Chatbot 生成；当前检索只使用 SQLite FTS5 和 API 重排，因此显卡占用为 0 是正常现象。
