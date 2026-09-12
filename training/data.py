import os
from collections import defaultdict
from datasets import Dataset, load_dataset, concatenate_datasets


def _take_by_fraction(indices, tokens, total: int, frac: float, label: str):
    target = int(total * frac)
    cumulative = 0
    selected = []
    truncate_info = None

    for idx, n in zip(indices, tokens):
        if cumulative + n > target:
            needed = target - cumulative
            selected.append(idx)
            truncate_info = (idx, needed, n)
            cumulative = target
            break
        cumulative += n
        selected.append(idx)
        if cumulative >= target:
            break

    print(f"  {label}: {cumulative:,}/{total:,} tokens")
    return selected, truncate_info


def _apply_truncations(dataset, selected, truncations):
    """Truncate last-selected docs by character ratio to hit exact token targets."""
    pos_to_trunc = {
        i: truncations[orig_idx]
        for i, orig_idx in enumerate(selected)
        if orig_idx in truncations
    }
    if not pos_to_trunc:
        return dataset

    data = {col: list(dataset[col]) for col in dataset.column_names}
    for pos, (needed, doc_total) in pos_to_trunc.items():
        char_frac = needed / doc_total if doc_total > 0 else 1.0
        data["text"][pos] = data["text"][pos][:int(len(data["text"][pos]) * char_frac)]
        data["num-tokens"][pos] = needed

    return Dataset.from_dict(data)


def get_dataset(args, parser):
    if args.dataset_fractions is not None:
        n = len(args.datasets.split())
        if len(args.dataset_fractions) != n:
            parser.error(
                f"--dataset_fractions has {len(args.dataset_fractions)} values "
                f"but --datasets has {n} datasets"
            )

    datasets = []
    for i, dataset_name in enumerate(args.datasets.split()):
        frac = args.dataset_fractions[i] if args.dataset_fractions is not None else args.dataset_fraction
        dataset = load_dataset(dataset_name, split="train", token=os.environ.get("HF_TOKEN"))

        dataset = dataset.shuffle(seed=args.data_seed)
        print(f"Shuffled {dataset_name} (seed={args.data_seed})")

        if frac >= 1.0:
            pass
        elif args.stratify_by:
            if args.stratify_by not in dataset.column_names:
                raise ValueError(
                    f"--stratify_by column '{args.stratify_by}' not found in {dataset_name}. "
                    f"Available: {dataset.column_names}"
                )
            strata = defaultdict(lambda: {"indices": [], "tokens": [], "total": 0})
            for idx, (cat, n) in enumerate(zip(dataset[args.stratify_by], dataset["num-tokens"])):
                strata[cat]["indices"].append(idx)
                strata[cat]["tokens"].append(n)
                strata[cat]["total"] += n
            print(f"Stratified sampling of {dataset_name} by '{args.stratify_by}' ({len(strata)} strata)")
            selected = []
            truncations = {}
            for cat in sorted(strata):
                sel, trunc = _take_by_fraction(
                    strata[cat]["indices"], strata[cat]["tokens"], strata[cat]["total"], frac, repr(cat)
                )
                selected.extend(sel)
                if trunc is not None:
                    orig_idx, needed, doc_total = trunc
                    truncations[orig_idx] = (needed, doc_total)
            dataset = dataset.select(selected)
            dataset = _apply_truncations(dataset, selected, truncations)
            print(f"Stratified result: {len(dataset):,} rows from {dataset_name}")
        else:
            tokens = dataset["num-tokens"]
            total = sum(tokens)
            indices = list(range(len(tokens)))
            selected, trunc = _take_by_fraction(indices, tokens, total, frac, dataset_name)
            dataset = dataset.select(selected)
            if trunc is not None:
                orig_idx, needed, doc_total = trunc
                dataset = _apply_truncations(dataset, selected, {orig_idx: (needed, doc_total)})

        datasets.append(dataset)
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def add_dataset_args(parser):
    """Add all dataset-related args to an argparse parser. Call from train.py and tokenizer.py."""
    g = parser.add_argument_group("data")
    g.add_argument("--datasets", type=str, default="",
                   help="Space-separated HF dataset name(s) to load")
    g.add_argument("--data_seed", type=int, default=42,
                   help="RNG seed for dataset shuffling and train/eval split")
    g.add_argument("--dataset_fraction", type=float, default=1.0,
                   help="Fraction of total tokens to use per dataset (e.g. 0.5)")
    g.add_argument("--dataset_fractions", type=float, nargs="+", default=None,
                   help="Per-dataset token fractions, one per dataset. Overrides --dataset_fraction.")
    g.add_argument("--stratify_by", type=str, default="category",
                   help="Column to stratify on when subsampling (e.g. 'category'); preserves source proportions. Set to '' to disable.")
