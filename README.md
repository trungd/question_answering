Implementations of question answering models

# Datasets

- SQuAD v1

# Models

- Dynamic Coattention Networks (DCN)
- QANet ([paper](https://arxiv.org/abs/1804.09541))
  - [SQuAD](./model_configs/squad_qanet.yml)
  
# Run

```
python -m dlex.train -c squad_qanet
```

# Results

|  | configs | EM  | F1  |
|--------|--------|---|---|
|QANet | squad_qanet.yml | 66.35 | 76.19 |
|DCN   | squad_dcn.yml   |        |     |