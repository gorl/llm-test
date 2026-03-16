from datasets import load_from_disk

def read_hf_text(path: str) -> str:
    ds = load_from_disk(path)
    return "\n".join(ds["text"])

def load_hf_dataset(path: str) -> list[str]:
    ds = load_from_disk(path)
    return ds["text"]