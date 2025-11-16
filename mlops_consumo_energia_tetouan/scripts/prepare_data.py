# scripts/prepare_data.py

from mlops_consumo_energia_tetouan.dataset import DataPreparer


def main():
    preparer = DataPreparer()
    preparer.run()


if __name__ == "__main__":
    main()