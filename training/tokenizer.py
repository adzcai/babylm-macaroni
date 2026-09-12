import os
import argparse
import tempfile
import yaml
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
from transformers import PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer
from data import get_dataset, add_dataset_args


def _write_chunk(texts, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in texts:
            if line.strip():
                f.write(line.rstrip("\n") + "\n")
    return path


def train_and_save_tokenizer(dataset, tokenizer_dir, vocab_size, min_frequency):
    tokenizer = ByteLevelBPETokenizer()

    texts = dataset["text"]
    n_workers = min(os.cpu_count() or 4, 16)
    chunk_size = max(1, len(texts) // n_workers)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = [os.path.join(tmpdir, f"chunk_{i}.txt") for i in range(len(chunks))]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(_write_chunk, chunks, paths))
        for i, path in enumerate(paths):
            mb = os.path.getsize(path) / 1e6
            print(f"chunk_{i}.txt: {mb:.1f} MB")

        # Preview for sanity check
        with open(paths[0], "r", encoding="utf-8") as f:
            for i, line in zip(range(3), f, strict=False):
                preview = line[:88].replace("\n", "\\n")
                print(f"[temp line {i+1}] {preview}")

        tokenizer.train(
            files=paths,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )

    tokenizer.save_model(str(tokenizer_dir))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    fast_tokenizer.save_pretrained(str(tokenizer_dir))

    return fast_tokenizer


def normalize_arg_types(args):
    float_keys = [
        "dataset_fraction",
    ]

    int_keys = [
        "vocab_size",
        "min_frequency",
        "data_seed",
    ]

    for key in float_keys:
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(args, key, float(getattr(args, key)))

    for key in int_keys:
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(args, key, int(getattr(args, key)))

    return args


def read_config(args, config):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return

    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML mapping of option names to values")

    allowed_keys = set(vars(args).keys())
    unknown_keys = sorted(set(cfg.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown config option(s): {', '.join(unknown_keys)}")

    for key, value in cfg.items():
        setattr(args, key, value)


def main():
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

    parser = argparse.ArgumentParser(description="Train and save a BPE tokenizer")
    parser.add_argument("--tokenizer_config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save the tokenizer")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size for tokenizer")
    parser.add_argument("--min_frequency", type=int, default=2, help="Minimum token frequency")
    add_dataset_args(parser)
    args = parser.parse_args()
    read_config(args, args.tokenizer_config)
    args = normalize_arg_types(args)

    if not args.output_dir:
        parser.error("output_dir must be set via --tokenizer_config or --output_dir")

    dataset = get_dataset(args, parser)

    tokenizer_dir = Path(args.output_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    train_and_save_tokenizer(dataset, tokenizer_dir, args.vocab_size, args.min_frequency)
    print(f"Tokenizer saved to {tokenizer_dir}")


if __name__ == "__main__":
    main()
