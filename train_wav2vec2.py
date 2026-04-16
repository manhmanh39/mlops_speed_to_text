# file: train_wav2vec2.py
"""
Training script (standalone) with MLflow experiment tracking.
Usage example:
python train_wav2vec2.py --extracted_dir /content/data_train_70.6 \\
    --meta_csv /content/data_train_70.6/metadata.csv \\
    --model_id nguyenvulebinh/wav2vec2-large-vi-vlsp2020 \\
    --output_dir wav2vec2-finetuned --mode train

Note: install required packages first (transformers, datasets,
huggingface_hub, torch, torchaudio, jiwer, safetensors, mlflow as needed).
"""

import argparse
import logging
import os
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from typing import Dict, List, Union

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset
from huggingface_hub import hf_hub_download, login
from transformers import Trainer, TrainingArguments, Wav2Vec2Processor

try:
    from safetensors.torch import save_file as save_safetensors
except Exception:
    save_safetensors = None


# configure root logger for simple output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defaults used when CLI args are not provided
EXTRACTED_DIR = "/content/data_train_70.6"
META_CSV = os.path.join(EXTRACTED_DIR, 'metadata.csv')
MODEL_ID_DEFAULT = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
OUTPUT_DIR_DEFAULT = "wav2vec2-finetuned"

# MLflow defaults
MLFLOW_TRACKING_URI_DEFAULT = "http://mlflow:5000"
MLFLOW_EXPERIMENT_DEFAULT = "wav2vec2-vietnamese"


# ─── MLflow helpers ──────────────────────────────────────────────────────────

def setup_mlflow(tracking_uri: str, experiment_name: str):
    """Configure MLflow tracking server and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow configured: uri=%s  experiment=%s",
        tracking_uri,
        experiment_name,
    )


def log_dvc_info(meta_csv: str, extracted_dir: str):
    """Log DVC-tracked dataset info as MLflow tags/params."""
    try:
        import subprocess
        result = subprocess.run(
            ["dvc", "status", "--json"],
            capture_output=True, text=True, cwd=extracted_dir
        )
        if result.returncode == 0:
            mlflow.set_tag("dvc.status", result.stdout.strip()[:500])
    except Exception:
        pass  # DVC not installed or not a DVC repo

    # Log dataset path for traceability
    mlflow.log_param("data.meta_csv", meta_csv)
    mlflow.log_param("data.extracted_dir", extracted_dir)

    # Log row count for quick sanity check
    try:
        meta = pd.read_csv(
            meta_csv, sep='|', header=None, names=['filename', 'transcript']
        )
        mlflow.log_param("data.num_rows", len(meta))
    except Exception:
        pass


# ─── MLflow Trainer callback ─────────────────────────────────────────────────

class MLflowTrainerCallback:
    """Minimal callback shim: logs Trainer metrics to active MLflow run."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)


# ─── Patched Trainer that streams metrics to MLflow ──────────────────────────

class MLflowTrainer(Trainer):
    """Trainer subclass that logs eval/train metrics to MLflow."""

    def log(self, logs: Dict, start_time=None):
        super().log(logs)
        step = self.state.global_step if self.state else 0
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                try:
                    mlflow.log_metric(key, value, step=step)
                except Exception:
                    pass


# Optional HF login helper: call login() when token provided
def maybe_login(hf_token: str):
    if hf_token:
        try:
            login(hf_token)
            logger.info("Logged into Hugging Face Hub.")
        except Exception as e:
            logger.warning("Failed to login to HF Hub: %s", e)


# Prepare datasets from extracted archive + metadata
def prepare_datasets(
    extracted_dir: str = EXTRACTED_DIR,
    meta_csv: str = META_CSV,
    wav_dir: str = None,
    # max_duration: float = 15.0  # GIỚI HẠN: Lọc bỏ file > 15 giây
):
    # default wav directory inside extracted_dir
    if wav_dir is None:
        wav_dir = os.path.join(extracted_dir, 'wavs')

    # ensure metadata CSV exists
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(
            f"Metadata file not found: {meta_csv}"
        )

    # read metadata: two columns (filename|transcript)
    meta = pd.read_csv(meta_csv, sep='|', header=None, names=['filename', 'transcript'])

    # SỬA Ở ĐÂY: KHÔNG xóa dấu cách. Chỉ xóa các ký tự đặc biệt gây nhiễu.
    # Pattern mới loại bỏ space khỏi danh sách bị xóa
    pattern = r"[\,\?\.\!\-\;\:\"\'\'\"\"\%\…]"
    meta['transcript'] = (
        meta['transcript']
        .str.lower()
        .str.replace(pattern, '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)  # Gộp nhiều space thành 1 space
        .str.strip()
    )

    # attach full audio path for each filename
    meta['audio_path'] = meta['filename'].apply(
        lambda f: os.path.join(wav_dir, f)
    )

    # create a Hugging Face dataset from the dataframe
    hf_ds = Dataset.from_pandas(meta[['audio_path', 'transcript']])

    # cast the audio path column to an Audio feature with sr=16k
    hf_ds = hf_ds.cast_column(
        'audio_path', Audio(sampling_rate=16_000)
    )

    # split into train/test (10% test)
    dsplits = hf_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = dsplits['train']
    eval_ds = dsplits['test']
    return train_ds, eval_ds


# Optional model loader: downloads model_handling.py if present
def load_model_and_processor(
    model_id: str = MODEL_ID_DEFAULT,
    device: str = None,
    trust_remote_code: bool = True,
):
    # try to fetch a custom model_handling.py from the hub repo
    try:
        model_script = hf_hub_download(
            repo_id=model_id, filename="model_handling.py"
        )
        model_loader = SourceFileLoader(
            "model_handling", model_script
        ).load_module()
    except Exception as e:
        logger.warning(
            "Could not download model_handling.py from the hub: %s. "
            "Proceeding without it.",
            e,
        )
        model_loader = None

    # load the processor (feature extractor + tokenizer)
    processor = Wav2Vec2Processor.from_pretrained(model_id)

    # if the repo provided a local model class, prefer it
    if model_loader is not None:
        ModelClass = getattr(model_loader, 'Wav2Vec2ForCTC', None)
        if ModelClass is not None:
            model = ModelClass.from_pretrained(
                model_id, trust_remote_code=trust_remote_code,
                ignore_mismatched_sizes=True  # <--- THÊM DÒNG NÀY
            )
        else:
            from transformers import AutoModelForCTC
            model = AutoModelForCTC.from_pretrained(model_id,
            ignore_mismatched_sizes=True ) # <--- THÊM DÒNG NÀY)
    else:
        from transformers import AutoModelForCTC
        model = AutoModelForCTC.from_pretrained(model_id, ignore_mismatched_sizes=True ) # <--- THÊM DÒNG NÀY)

    # move model to selected device (cuda if available)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device


# Data collator that pads inputs and label sequences for CTC training
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        inputs = [{"input_values": f["input_values"]} for f in features]
        labels = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(
            inputs, padding=self.padding, return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=self.padding,
            return_tensors="pt",
        )["input_ids"]
        labels_batch = labels_batch.masked_fill(
            labels_batch == self.processor.tokenizer.pad_token_id,
            -100,
        )

        batch["labels"] = labels_batch
        return batch


# compute_metrics function used by Trainer to compute WER on eval
def compute_metrics(pred, processor_ref):
    from jiwer import wer
    pred_ids = np.argmax(pred.predictions, axis=-1)
    label_ids = np.where(
        pred.label_ids != -100,
        pred.label_ids,
        processor_ref.tokenizer.pad_token_id,
    )
    pred_str = processor_ref.batch_decode(
        pred_ids, group_tokens=True, skip_special_tokens=True
    )
    label_str = processor_ref.batch_decode(
        label_ids, group_tokens=True, skip_special_tokens=True
    )
    wer_score = wer(label_str, pred_str)
    return {"wer": wer_score}


def align_model_and_tokenizer(model, processor):
    tokenizer_size = len(processor.tokenizer)
    model_vocab = getattr(model.config, "vocab_size", None)
    logger.info(
        "Model vocab_size=%s   Tokenizer size=%s",
        model_vocab,
        tokenizer_size,
    )

    if model_vocab != tokenizer_size:
        logger.info(
            "Resizing model embeddings: %s -> %s",
            model_vocab,
            tokenizer_size,
        )
        try:
            model.resize_token_embeddings(tokenizer_size)
        except Exception as e:
            logger.warning("resize_token_embeddings failed: %s", e)
        model.config.vocab_size = tokenizer_size

    if (
        getattr(model.config, "pad_token_id", None)
        != processor.tokenizer.pad_token_id
    ):
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        logger.info(
            "Set model.config.pad_token_id = %s",
            model.config.pad_token_id,
        )


def build_training_args(
    output_dir: str,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: int,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        # overwrite_output_dir=True,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.15,
        weight_decay=0.01,
        num_train_epochs=num_train_epochs,
        eval_steps=500,
        save_steps=500,
        logging_steps=200,
        eval_strategy="steps",
        save_total_limit=2,
        gradient_checkpointing=True,
        fp16=True,
        dataloader_num_workers=0,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        # Disable built-in W&B / other integrations; we use MLflow directly
        report_to="none",
    )


# preprocessing for a single dataset example
def prepare_batch(batch, processor_ref):
    arr = batch['audio_path']['array']
    sr = batch['audio_path']['sampling_rate']

    inp = processor_ref.feature_extractor(
        arr, sampling_rate=sr, return_tensors='np'
    ).input_values[0]
    inp = np.asarray(inp, dtype=np.float32)

    transcript = batch['transcript']
    word_delimiter = processor_ref.tokenizer.word_delimiter_token
    if word_delimiter is not None:
        transcript = transcript.replace(" ", word_delimiter)

    lbl = processor_ref.tokenizer(transcript).input_ids
    return {'input_values': inp, 'labels': lbl}


# Run end-to-end training with sensible defaults
def run_training( # noqa: C901
    extracted_dir: str = EXTRACTED_DIR,
    meta_csv: str = META_CSV,
    model_id: str = MODEL_ID_DEFAULT,
    output_dir: str = OUTPUT_DIR_DEFAULT,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 6,
    learning_rate: float = 1e-5,
    num_train_epochs: int = 7,
    hf_token: str = None,
    push_to_hub: bool = False,
    repo_id: str = None,
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI_DEFAULT,
    mlflow_experiment: str = MLFLOW_EXPERIMENT_DEFAULT,
):
    # ── MLflow setup ─────────────────────────────────────────────────────────
    setup_mlflow(mlflow_tracking_uri, mlflow_experiment)

    with mlflow.start_run() as run:
        logger.info("MLflow run_id: %s", run.info.run_id)

        # ── Log DVC data provenance ───────────────────────────────────────────
        log_dvc_info(meta_csv, extracted_dir)

        # ── Log all hyper-parameters ──────────────────────────────────────────
        mlflow.log_params({
            "model_id": model_id,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": (
                per_device_train_batch_size * gradient_accumulation_steps
            ),
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "output_dir": output_dir,
        })

        # ── Prepare datasets ──────────────────────────────────────────────────
        train_ds, eval_ds = prepare_datasets(extracted_dir, meta_csv)
        mlflow.log_params({
            "train_size": len(train_ds),
            "eval_size": len(eval_ds),
        })

        # ── Load model + processor ────────────────────────────────────────────
        model, processor, device = load_model_and_processor(model_id)
        align_model_and_tokenizer(model, processor)
        mlflow.set_tag("device", device)

        data_collator = DataCollatorCTCWithPadding(processor=processor)

        train_prepped = train_ds.map(
            lambda b: prepare_batch(b, processor),
            remove_columns=train_ds.column_names,
        )
        eval_prepped = eval_ds.map(
            lambda b: prepare_batch(b, processor),
            remove_columns=eval_ds.column_names,
        )

        training_args = build_training_args(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
        )


        from transformers import EarlyStoppingCallback
        # ── Use patched MLflowTrainer ─────────────────────────────────────────
        trainer = MLflowTrainer(
            model=model,
            args=training_args,
            train_dataset=train_prepped,
            eval_dataset=eval_prepped,
            data_collator=data_collator,
            compute_metrics=lambda pred: compute_metrics(pred, processor),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Rất quan trọng!
        )

        resume_from_checkpoint = None
        if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
            resume_from_checkpoint = False # Trainer sẽ tự tìm checkpoint mới nhất trong output_dir

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        eval_results = trainer.evaluate()

        # ── Log final eval metrics ────────────────────────────────────────────
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"final_{key}", value)

        # ── Save model artifacts ──────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)

        try:
            trainer.save_model(output_dir)
            processor.save_pretrained(output_dir)
            logger.info("Saved model and processor to %s", output_dir)
        except Exception as e:
            logger.exception(
                "Failed to save model/processor with save_pretrained: %s", e
            )

        # Save safetensors checkpoint
        try:
            if save_safetensors is not None:
                sd = {k: v.cpu() for k, v in model.state_dict().items()}
                safetensors_path = os.path.join(
                    output_dir, "model.safetensors"
                )
                save_safetensors(sd, safetensors_path)
                logger.info("Saved safetensors to %s", safetensors_path)
            else:
                logger.warning(
                    "safetensors not installed; skipping model.safetensors save."
                )
        except Exception as e:
            logger.exception("Failed to write safetensors: %s", e)

        # ── Log model directory as MLflow artifact ────────────────────────────
        try:
            mlflow.log_artifacts(output_dir, artifact_path="model")
            logger.info("Logged model artifacts to MLflow.")
        except Exception as e:
            logger.warning("Could not log artifacts to MLflow: %s", e)

        # ── Optionally push to Hugging Face Hub ──────────────────────────────
        if push_to_hub and repo_id:
            try:
                logger.info("Pushing model to Hugging Face Hub: %s", repo_id)
                trainer.push_to_hub(repo_id=repo_id, private=False)
                processor.push_to_hub(repo_id=repo_id, private=False)
                mlflow.set_tag("hf_repo_id", repo_id)
                logger.info(
                    "Successfully pushed model and processor to Hub: %s", repo_id
                )
            except Exception as e:
                logger.exception("Failed to push model to Hub: %s", e)
        else:
            if push_to_hub:
                logger.warning(
                    "push_to_hub enabled but repo_id not provided; skipping."
                )

        # ── Final WER summary ─────────────────────────────────────────────────
        try:
            wer_val = eval_results['eval_wer']
            logger.info("Final eval WER: %.4f", wer_val)
        except Exception:
            logger.info(
                "Evaluation finished; check trainer.evaluate() results."
            )


def main():
    parser = argparse.ArgumentParser(
        description='Training script with MLflow tracking and DVC data versioning.'
    )

    # NOTE FOR USERS: default setup (batch 8 with
    # gradient_accumulation_steps 4) may consume ~13.5GB VRAM on T4.
    # Reduce per_device_train_batch_size or gradient_accumulation_steps
    # if you run out of memory.

    parser.add_argument(
        '--extracted_dir', type=str, default=EXTRACTED_DIR,
        help=(
            'Directory containing extracted dataset; expected to '
            'include a "wavs/" subfolder with audio files.'
        ),
    )
    parser.add_argument(
        '--meta_csv', type=str, default=META_CSV,
        help=(
            'Path to the metadata CSV (format: filename|transcript) '
            'used to build the HF Dataset.'
        ),
    )
    parser.add_argument(
        '--model_id', type=str, default=MODEL_ID_DEFAULT,
        help=(
            'Hugging Face model repo id or local path for the '
            'pretrained Wav2Vec2 model/processor to load.'
        ),
    )
    parser.add_argument(
        '--output_dir', type=str, default=OUTPUT_DIR_DEFAULT,
        help=(
            'Directory where the fine-tuned model, processor, '
            'and optional safetensors will be saved.'
        ),
    )
    parser.add_argument(
        '--hf_token', type=str, default=None,
        help=(
            '(Optional) Hugging Face access token used to login '
            'and interact with the Hub.'
        ),
    )
    parser.add_argument(
        '--per_device_train_batch_size', type=int, default=8,
        help=(
            'Per-device training batch size. Reduce this if you '
            'encounter out-of-memory (OOM) errors on your GPU.'
        ),
    )
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=6,
        help=(
            'Number of steps to accumulate gradients before updating '
            'model weights.'
        ),
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-5,
        help='Initial learning rate for the optimizer.',
    )
    parser.add_argument(
        '--num_train_epochs', type=int, default=7,
        help='Total number of training epochs to run.',
    )
    parser.add_argument(
        '--push_to_hub', action='store_true',
        help='Push the fine-tuned model to Hugging Face Hub.',
    )
    parser.add_argument(
        '--repo_id', type=str, default=None,
        help=(
            'Hugging Face Hub repository ID for pushing the model. '
            'Required if --push_to_hub is enabled.'
        ),
    )
    # ── MLflow arguments ──────────────────────────────────────────────────────
    parser.add_argument(
        '--mlflow_tracking_uri', type=str,
        default=os.environ.get(
            "MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI_DEFAULT
        ),
        help='MLflow tracking server URI.',
    )
    parser.add_argument(
        '--mlflow_experiment', type=str,
        default=os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_DEFAULT
        ),
        help='MLflow experiment name.',
    )

    args = parser.parse_args()
    maybe_login(args.hf_token)

    run_training(
        extracted_dir=args.extracted_dir,
        meta_csv=args.meta_csv,
        model_id=args.model_id,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        hf_token=args.hf_token,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
    )


if __name__ == '__main__':
    main()
