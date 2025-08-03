# Micro LLM
This project demonstrates the training and inference of a small language model using PyTorch. It includes utilities for training, visualization, and interactive chat capabilities.
```
micro-llm-project/
├── src/
│   ├── micro_llm.py
│   ├── training_utils.py
│   ├── train_with_visualization.py
│   └── config.py
├── notebooks/
│   └── model_exploration.ipynb
├── visualizations/
│   └── index.html
├── data/
├── models/
├── training_logs/
├── requirements.txt
└── README.md
```
## Installation
To set up the environment, install the required packages using uv:
uv sync

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126  

