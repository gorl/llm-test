from datasets import load_dataset, concatenate_datasets

# Загружаем части Taiga
# proza = load_dataset("cointegrated/taiga_stripped_proza", split="train")

rest_splits = [
    "Arzamas",
    "Fontanka",
    "Interfax",
    "KP",
    "Lenta",
    "Magazines",
    "NPlus1",
    "Subtitles",
    "social",
]

rest_parts = [
    load_dataset("cointegrated/taiga_stripped_rest", split=split)
    for split in rest_splits
]

rest = concatenate_datasets(rest_parts)
# Объединяем
# ds = concatenate_datasets([proza, rest])
ds = rest
print("Total rows:", len(ds))

# Taiga не перемешана → перемешаем
ds = ds.shuffle(seed=42)

# Аналогичные сплиты
test  = ds.select(range(0, 10_000))
val   = ds.select(range(10_000, 20_000))
train = ds.select(range(20_000, len(ds)))


train_100 = ds.select(range(0, 100_000))
train_500 = ds.select(range(0, 500_000))

# Сохраняем
test.save_to_disk("./data/taiga_test")
val.save_to_disk("./data/taiga_val")
train.save_to_disk("./data/taiga_train")
train_100.save_to_disk("./data/taiga_train_100")
train_500.save_to_disk("./data/taiga_train_500")