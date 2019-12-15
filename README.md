Implementations of question answering models

# Datasets

- SQuAD v1
- SQuAD v2

# Models

- Dynamic Coattention Networks (DCN) ([paper](https://arxiv.org/abs/1611.01604))
- QANet ([paper](https://arxiv.org/abs/1804.09541))
  
# Run

```
pip install dlex
```

```
# Training
python -m dlex.train -c squad_qanet
# Evaluating
python -m dlex.evaluate -c squad_qanet -l valid-best-em
```

# Results

|  | configs | EM  | F1  |
|--------|--------|---|---|
|QANet | squad_qanet.yml | 66.35 | 76.19 |
|DCN   | squad_dcn.yml   |        |     |