from datasets import load_dataset
ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train")

# Suggested default splits (data is pre-shuffled):
#   rows 0..9,999       -> test  (10K stories)
#   rows 10,000..19,999 -> val   (10K stories)
#   rows 20,000..end    -> train (2,712,634 stories)
test  = ds.select(range(0, 10_000))
val   = ds.select(range(10_000, 20_000))
train = ds.select(range(20_000, len(ds)))
train_100 = ds.select(range(0, 100_000))
train_500 = ds.select(range(0, 500_000))


test.save_to_disk("~/data/tinystories_test")
val.save_to_disk("~/data/tinystories_val")
train.save_to_disk("~/data/tinystories_train")
train_100.save_to_disk("~/data/tinystories_train_100")
train_500.save_to_disk("~/data/tinystories_train_500")

# scp -O -i ~/.ssh/id_ed25519 -r ../llm_baseline_project r6qyi5vx0ub9yh-644112d4@ssh.runpod.io:/workspace/