# dsa4213-ay2526-group41

## Dataset

[MS MARCO](https://microsoft.github.io/msmarco/Datasets.html)

## Notes

- May have queries where there are no relevant passages given that the `qrels.dev.tsv` and the `qrels.train.tsv` is smaller than `queries.dev.tsv` and `queries.train.tsv`.

## Getting Started

1. Clone Repository

    ```bash
    cd <desired-directory>
    git clone https://github.com/zzhenjie01/dsa4213-assignments.git
    ```

2. Install Python Packages

    We are using `uv` as our Python installation and package manager as it is extremely fast and resolves dependencies better that other package manager such as `pip`. [uv installation link](https://docs.astral.sh/uv/getting-started/installation/)

    Use PyTorch CPU-only (For Devices without NVIDIA GPU, including Macs):

    ```bash
    uv sync --extra cpu
    ```

    For devices with NVIDIA GPU:

    ```bash
    uv sync --extra cu128
    ```

## Docker Setup

```bash
cd docker
```

```bash
docker compose --project-name mab-retrieval up -d
```

## Data Preprocessing

- Run `src/data_preprocessing_msmarco.ipynb`
- Run `src/data_preprocessing_nq.ipynb`

## Data Ingestion

```bash
python src/data_ingestion_msmarco.py
```

```bash
python src/data_ingestion_nq.py
```

## MAB Training

### Static Baseline Retrievals

**MS MARCO:**

```bash
python src/run_static_baselines.py --metric recall --dataset_name ms_marco
```

**Natural Questions:**

```bash
python src/run_static_baselines.py --metric recall --dataset_name nq
```

### Epsilon Greedy Algorithm

**MS MARCO:**

```bash
python src/train_epsilon_greedy.py --epsilons 0.01 0.1 0.3 --lambda_param 0.5 --metric recall --seed 42 --dataset_name ms_marco
```

```bash
python src/train_epsilon_greedy.py --epsilons 0.1 --lambda_param 0.1 --metric recall --seed 42 --dataset_name ms_marco
```

```bash
python src/train_epsilon_greedy.py --epsilons 0.1 --lambda_param 0.9 --metric recall --seed 42 --dataset_name ms_marco
```

**Natural Questions:**

```bash
python src/train_epsilon_greedy.py --epsilons 0.01 0.1 0.3 --lambda_param 0.5 --metric recall --seed 42 --dataset_name nq
```

```bash
python src/train_epsilon_greedy.py --epsilons 0.1 --lambda_param 0.1 --metric recall --seed 42 --dataset_name nq
```

```bash
python src/train_epsilon_greedy.py --epsilons 0.1 --lambda_param 0.9 --metric recall --seed 42 --dataset_name nq
```

### UCB Algorithm

**MS MARCO:**

```bash
python src/train_ucb.py --c_values 1.0 2.0 3.0 --lambda_param 0.5 --metric recall --seed 42 --dataset_name ms_marco
```

```bash
python src/train_ucb.py --c_values 2.0 --lambda_param 0.1 --metric recall --seed 42 --dataset_name ms_marco
```

```bash
python src/train_ucb.py --c_values 2.0 --lambda_param 0.9 --metric recall --seed 42 --dataset_name ms_marco
```

**Natural Questions:**

```bash
python src/train_ucb.py --c_values 1.0 2.0 3.0 --lambda_param 0.5 --metric recall --seed 42 --dataset_name nq
```

```bash
python src/train_ucb.py --c_values 2.0 --lambda_param 0.1 --metric recall --seed 42 --dataset_name nq
```

```bash
python src/train_ucb.py --c_values 2.0 --lambda_param 0.9 --metric recall --seed 42 --dataset_name nq
```

### Thompson Sampling

**MS MARCO:**

```bash
python src/train_thompson.py --lambda_param 0.5 --metric recall --seed 42 --dataset_name ms_marco
```

```bash
python src/train_thompson.py --lambda_param 0.1 --metric recall --seed 42 --dataset_name ms_marco
```

```bash
python src/train_thompson.py --lambda_param 0.9 --metric recall --seed 42 --dataset_name ms_marco
```

**Natural Questions:**

```bash
python src/train_thompson.py --lambda_param 0.5 --metric recall --seed 42 --dataset_name nq
```

```bash
python src/train_thompson.py --lambda_param 0.1 --metric recall --seed 42 --dataset_name nq
```

```bash
python src/train_thompson.py --lambda_param 0.9 --metric recall --seed 42 --dataset_name nq
```