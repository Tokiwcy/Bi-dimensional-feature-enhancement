# Bi-dimensional-feature-enhancement

房价预测实验脚本集合，包含以下数据集与脚本：

- `Ames.py`
- `cal.py` (California Housing)
- `boston.py`
- `melb.py` (Melbourne Housing)
- `brazilian.py` (Brazilian Houses)

## 项目结构

- `data/`：所有数据文件
  - `train.csv`
  - `test.csv`
  - `ground_truth.csv`
  - `california_housing.csv`
  - `boston_housing_dataset.csv`
  - `melb_data.csv`
  - `brazilian_houses.arff`

## 环境安装

建议 Python `3.10+`，先创建虚拟环境，再安装依赖：

```bash
pip install -r requirements.txt
```

如果你要使用 embedding 功能，还需要配置 OpenAI Key：

```bash
set OPENAI_API_KEY=your_api_key
```

Linux / macOS:

```bash
export OPENAI_API_KEY=your_api_key
```

## 运行示例

### Ames

```bash
python Ames.py --mode baseline
python Ames.py --mode rag --rag-k 6,8,10 --seed 0,1,2,3,4
python Ames.py --mode emb_with_rag --rag-template compare --seed 0
```

### California

```bash
python cal.py --mode baseline,rag --rag-mode hybrid --rag-k 6 --seed 0,1,2,3,4
```

### Boston

```bash
python boston.py --mode all --seed 0,1,2,3,4
```

### Melbourne

```bash
python melb.py --mode all --seed 42
```

### Brazilian

```bash
python brazilian.py --mode all --seed 0,1,2,3,4
```

## 说明

- 当前脚本默认从 `data/` 目录读取数据。
- 不同脚本参数略有差异，请使用 `python xxx.py --help` 查看完整参数。
- `openai` 与 `tabpfn` 为可选依赖，未安装时对应功能会自动跳过。
