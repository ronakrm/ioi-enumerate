# Enumerating Some Basic Runs of the Original GPT-2 Small IOI Result
This repo just runs the IOI task on a large number of the task templates used in the original paper.
The goal is to demonstrate that there are "structured subsets" of names that always fail to properly IOI,
even under the most generous conditions. Take a look at the notebook for some results!

# Use

## Install
```
poetry install
poetry run python -m spacy download en_core_web_sm
```

## Run
```
poetry shell
python get_logits.py
```

Plot and analyze with the notebook `plot_results.ipynb`.
