import os
import argparse
import yaml
from dotenv import load_dotenv
import json
from pathlib import Path
from datetime import datetime
from data import get_dataset, add_dataset_args
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
import torch


torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SaveMetadataCallback(TrainerCallback):
    """Writes the WandB run ID and SLURM job ID to output_dir at training start."""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        import wandb
        if wandb.run is not None:
            (self.output_dir / "wandb_run_id.txt").write_text(wandb.run.id)
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            (self.output_dir / "slurm_job_id.txt").write_text(slurm_job_id)


class SaveEpochSchedulerCallback(TrainerCallback): # todo: put this in a separate file if it grows more complex
    """Saves checkpoints at a user-defined schedule of epoch values.

    Args:
        schedule: Either a float (uniform interval) or a sorted list of epoch
                  values. If a list, must contain at least 2 entries.

    Examples:
        # Uniform interval: 0.5, 1.0, 1.5, 2.0, ...
        SaveEpochSchedulerCallback(0.5)

        # Dense early, sparse later: 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, ...
        SaveEpochSchedulerCallback([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4])

        # Sparse early, dense later: 1, 2, 4, 4.5, 5, 5.5, 6, 6.5, ...
        SaveEpochSchedulerCallback([1, 2, 4, 4.5, 5, 5.5, 6])
    """
    def __init__(self, schedule: list[float] = [1.0]):
        if len(schedule) == 1:
            # Uniform mode: no explicit checkpointing schedule, just a fixed interval forever.
            self.schedule = []
            self.tail_interval = schedule[0]
        elif len(schedule) >= 2:
            self.schedule = sorted(schedule)
            # Infer the tail interval from the gap between the last two entries.
            self.tail_interval = self.schedule[-1] - self.schedule[-2]
        else:
            raise ValueError("Empty epoch schedule is not allowed.")
        self._reset()

    def _reset(self):
        self._idx = 0
        # If schedule is empty (uniform mode), start at the tail interval directly.
        self._next_save = self.schedule[0] if self.schedule else self.tail_interval

    def _advance(self):
        """Move to the next save point, using the tail interval once the schedule is exhausted."""
        self._idx += 1
        if self._idx < len(self.schedule):
            self._next_save = self.schedule[self._idx]
        else:
            self._next_save += self.tail_interval

    def on_train_begin(self, args, state, control, **kwargs):
        """Resets the first save point, useful if the same callback object is reused"""
        self._reset()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.epoch is None:  # do nothing if epoch is not being tracked
            return control

        eps = 1e-12  # small tolerance for floating point comparisons

        if state.epoch + eps >= self._next_save:
            control.should_save = True  # ask Trainer to perform a checkpoint save

            # We use a while loop instead of just adding once in case
            # training jumps past more than one boundary between callback calls.
            while state.epoch + eps >= self._next_save:
                self._advance()

        return control


def get_context_length(model_config):
    for attr in ["n_ctx", "max_position_embeddings", "n_positions"]:
        value = getattr(model_config, attr, None)
        if value is not None:
            return value

    raise ValueError(
        "Could not infer context length from config. "
        "Expected one of: n_ctx, max_position_embeddings, n_positions."
    )


def tokenize_and_chunk(batch, tokenizer, max_length):
    texts = [t + tokenizer.eos_token for t in batch["text"]]
    tokenized = tokenizer(texts, truncation=False, padding=False)
    concatenated = {k: sum(tokenized[k], []) for k in tokenized}
    total = (len(concatenated["input_ids"]) // max_length) * max_length
    return {k: [v[i:i + max_length] for i in range(0, total, max_length)] for k, v in concatenated.items()}


# metadata keys that should not be passed to the training script as args
NON_CONFIG_KEYS = {
    "best_eval_loss",
}

def read_config(args, config):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
        
    if cfg is None:
        return

    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML mapping of option names to values")
    config_keys = set(cfg.keys())
    allowed_keys = set(vars(args).keys())
    unknown_keys = sorted(config_keys - allowed_keys - NON_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Unknown config option(s): {', '.join(unknown_keys)}"
        )

    # YAML values fill argparse args, except certain metadata keys
    for key, value in cfg.items():
        if key in NON_CONFIG_KEYS:
            continue

        setattr(args, key, value)


def normalize_arg_types(args):
    float_keys = [
        "dataset_fraction",

        "eval_intervals",

        "learning_rate",
        "min_lr_rate",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "max_grad_norm",
        "warmup_ratio",
    ]

    int_keys = [
        "num_workers",
        "data_seed",
        "model_seed",

        "batch_size",
        "gradient_accumulation_steps",
        "epochs",
    ]

    for key in float_keys:
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(args, key, float(getattr(args, key)))

    for key in int_keys:
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(args, key, int(getattr(args, key)))

    return args


def main():
    # Search for .env from this file's directory upward
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

    for var in ("WANDB_ENTITY", "WANDB_PROJECT"):
        if not os.environ.get(var):
            raise EnvironmentError(f"Required environment variable {var} is not set")

    parser = argparse.ArgumentParser(description="Train a Causal Language Model from scratch")

    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    data = parser.add_argument_group("data")
    data.add_argument("--tokenizer_dir", type=str, default="", help="Directory containing a saved tokenizer (run tokenizer.py first)")
    data.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading")
    add_dataset_args(data)

    model = parser.add_argument_group("model")
    model.add_argument("--model_config", type=str, default="small_config.json", help="Path to model config json file")
    model.add_argument("--checkpoint_dir", type=str, default="", help="Directory to load model checkpoint from")
    model.add_argument("--init_new_model", type=str, choices=["True", "False"], default="False", help="Initialize a new model from config even when checkpoint_dir is set (tokenizer is still loaded from tokenizer_dir)")
    model.add_argument("--resume_from_checkpoint", type=str, default="", help="Checkpoint directory to resume training from (restores model weights, optimizer, scheduler, and RNG state)")
    model.add_argument("--model_seed", type=int, default=42, help="Random seed for model and optimizer initialization")

    training = parser.add_argument_group("training")
    training.add_argument("--learning_rate", type=float, default=1e-04, help="Learning rate")
    training.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr", help="Learning rate scheduler type")
    training.add_argument("--min_lr_rate", type=float, default=0.1, help="Minimum learning rate")
    training.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    training.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    training.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    training.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam epsilon")
    training.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    training.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    training.add_argument("--batch_size", type=int, default=64, help="Training batch size per device")
    training.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    training.add_argument("--epochs", type=int, default=3, help="Number of training epochs")    

    ckpt_eval = parser.add_argument_group("checkpointing and evaluation")
    ckpt_eval.add_argument("--save_intervals", type=float, nargs="+", default=[500], help="Interval(s) for saving checkpoints. Single value: uniform interval (int for steps, float for epoch ratio). Multiple values: explicit epoch schedule")
    ckpt_eval.add_argument("--eval_strategy", choices=["steps", "epoch", "no"], default="epoch", help="Evaluation strategy")
    ckpt_eval.add_argument("--eval_intervals", type=float, default=1.0, help="Interval for evaluation. Integer number for steps, float for ratio of epoch(s)")

    output = parser.add_argument_group("output")
    output.add_argument("--output_dir", type=str, default="./output", help="Directory to save model outputs")
    output.add_argument("--model_name", type=str, default="", help="Hugging Face model repo name (e.g., user/model)")
    output.add_argument("--hf_usrname", type=str, default="johndoe", help="Hugging Face username (for pushing to hub)")
    output.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")

    args = parser.parse_args()

    read_config(args, args.config)

    args = normalize_arg_types(args)

    now = datetime.now()

    # Format the datetime object as a string without colons
    formatted_date = now.strftime("%Y%m%d_%H%M%S")

    border_line = "=" * 100

    tokenizer_dir = Path(args.tokenizer_dir)
    if not (tokenizer_dir / "tokenizer.json").exists():
        raise ValueError(f"No tokenizer found at {tokenizer_dir}. Run tokenizer.py first.")
    print(f"✅ Loading tokenizer from {tokenizer_dir}\n{border_line}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))

    print(f"🧹 Loading model dataset...\n{border_line}")
    dataset = get_dataset(args, parser)

    print(f"🧹 Loading model config...\n{border_line}")
    config = AutoConfig.from_pretrained(args.model_config)

    context_length = get_context_length(config)

    print(f"🧹 Tokenizing model dataset...\n{border_line}")
    tokenized_dataset = dataset.map(
        lambda batch: tokenize_and_chunk(batch, tokenizer, context_length),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=min(os.cpu_count() or 4, 16),
    )

    set_seed(args.model_seed)

    if args.checkpoint_dir and args.init_new_model == "False":
        print(f"🔧 Loading model from checkpoint {args.checkpoint_dir}...\n{border_line}")
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_dir)
    else:
        print(f"🔧 Initializing model...\n{border_line}")
        config.vocab_size = len(tokenizer)
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id        
        config._name_or_path = ""
        model = AutoModelForCausalLM.from_config(config)

    assert len(tokenizer) == model.config.vocab_size, f"Tokenizer vocab size {len(tokenizer)} and model config vocab size {model.config.vocab_size} must match"

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print(f"Model has {model.num_parameters()/1e6:.2f}M parameters")
    print(f"Model has {model.config.max_position_embeddings} max position embeddings")
    print(f"Model has {model.config.vocab_size} vocab size, meaning {model.lm_head.out_features} output features")
    print(args)
    lr_scheduler_kwargs = {}
    if args.lr_scheduler_type == "cosine_with_min_lr":
        lr_scheduler_kwargs = {"min_lr_rate": args.min_lr_rate}

    training_args = TrainingArguments(
        bf16=True,
        seed=args.model_seed,
        dataloader_num_workers=args.num_workers,
        hub_model_id=f"{args.hf_usrname}/{args.model_name}",
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        logging_dir=os.path.join(args.output_dir, "logs"),
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        remove_unused_columns=False,
        save_strategy="no",  # see SaveEpochSchedulerCallback
        save_total_limit=None,  # keep all checkpoints
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_intervals,
        report_to=["wandb"],
        run_name=f"{args.model_name}_{formatted_date}",
    )

    callbacks = [SaveMetadataCallback(args.output_dir), SaveEpochSchedulerCallback(args.save_intervals)]

    splits = tokenized_dataset.train_test_split(test_size=0.05, seed=args.data_seed)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    if not args.resume_from_checkpoint:
        print("💾 Saving tokenizer and initial model ...")
        checkpoint0_dir = os.path.join(args.output_dir, "checkpoint-0")
        trainer.save_model(checkpoint0_dir)

    print(f"🚀 Starting training...\n{border_line}")
    resume_ckpt = args.resume_from_checkpoint or None
    trainer.train(resume_from_checkpoint=resume_ckpt)
    final_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Final eval loss: {final_metrics['eval_loss']:.6f}")


    with open(Path(args.output_dir) / "sweep_config.json", "w") as f:
        json.dump(
            {
                "final_eval_loss": final_metrics["eval_loss"],
                "hyperparameters": {
                    "learning_rate": args.learning_rate,
                    "lr_scheduler_type": args.lr_scheduler_type,
                    "min_lr_rate": args.min_lr_rate,
                    "weight_decay": args.weight_decay,
                    "adam_beta1": args.adam_beta1,
                    "adam_beta2": args.adam_beta2,
                    "adam_epsilon": args.adam_epsilon,
                    "max_grad_norm": args.max_grad_norm,
                    "warmup_ratio": args.warmup_ratio,
                    "global_batch_size": args.batch_size * args.gradient_accumulation_steps,
                    "epochs": args.epochs,
                }
            },
            f,
            indent=2,
        )

    if args.push_to_hub:
        print(f"☁️ Pushing to Hugging Face Hub...\n{border_line}")
        trainer.push_to_hub(args.model_name)


if __name__ == "__main__":
    main()
