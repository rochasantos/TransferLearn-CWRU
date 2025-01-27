def get_dataset(dataset_name):
    from datasets import CWRU, Hust, UORED, Paderborn
    return {"CWRU": CWRU, "Hust": Hust, "UORED": UORED, "Paderborn": Paderborn}[dataset_name]