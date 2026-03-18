# Bi-dimensional-feature-enhancement

This repository contains house-price prediction experiment scripts and datasets:

- `Ames.py`
- `cal.py` (California Housing)
- `boston.py`
- `melb.py` (Melbourne Housing)
- `brazilian.py` (Brazilian Houses)

## Project Structure

- `data/`: all dataset files
  - `train.csv`
  - `test.csv`
  - `ground_truth.csv`
  - `california_housing.csv`
  - `boston_housing_dataset.csv`
  - `melb_data.csv`
  - `brazilian_houses.arff`

## Installation

Use Python `3.10+`. It is recommended to create a virtual environment first, then install dependencies:

```bash
pip install -r requirements.txt
```

If you want to use embedding features, configure your OpenAI API key:

```bash
set OPENAI_API_KEY=your_api_key
```

Linux / macOS:

```bash
export OPENAI_API_KEY=your_api_key
```

## Quick Start

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

## Notes

- Scripts default to reading datasets from the `data/` directory.
- CLI options differ slightly across scripts. Use `python <script>.py --help` for full arguments.
- `openai` and `tabpfn` are optional dependencies; related features are skipped if unavailable.
