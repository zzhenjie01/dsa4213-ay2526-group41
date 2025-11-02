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