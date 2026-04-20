# file: eval_wav2vec2.py
"""
Evaluation script that loads model architecture from hub (model_id)
AND local safetensors (model.safetensors) to ensure tokenizer <->
weights match the architecture used during finetune.

Experiment matrix:
  E1  Baseline Base    → base model, no finetune, no preproc
  E2  Baseline Large   → large model, no finetune, no preproc
  E3  Finetune Only    → large model, finetuned, no preproc
  E4  Finetune+DataEng → large model, finetuned, preproc
"""
import argparse
import csv
import glob
import logging
import os
import re
import uuid
from importlib.machinery import SourceFileLoader

import mlflow
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2Processor

# optional libraries (graceful fallback)
try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None
try:
    import noisereduce as nr
except Exception:
    nr = None
try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None
try:
    from scipy.io import wavfile
except Exception:
    wavfile = None
try:
    import librosa
except Exception:
    librosa = None
try:
    from unidecode import unidecode
except Exception:
    unidecode = None
try:
    from jiwer import cer, wer
except Exception:
    wer = cer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID_DEFAULT = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"

MLFLOW_TRACKING_URI_DEFAULT = "http://mlflow:5000"
MLFLOW_EXPERIMENT_DEFAULT = "wav2vec2-vietnamese-eval"


# ─── MLflow helpers ───────────────────────────────────────────────────────────
def setup_mlflow(tracking_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow configured: uri=%s  experiment=%s", tracking_uri, experiment_name)


# ─── Audio helpers ────────────────────────────────────────────────────────────
def normalize_audio_pydub(input_file, output_file, target_level=-24):
    if AudioSegment is None:
        raise RuntimeError("pydub not available")
    audio = (
        AudioSegment.silent(duration=500)
        + AudioSegment.from_file(input_file)
        + AudioSegment.silent(duration=500)
    )
    normalized = audio.apply_gain(target_level - audio.dBFS)
    normalized.export(output_file, format=output_file.split(".")[-1])


def remove_noise(input_file, output_file):
    if wavfile is None or nr is None:
        raise RuntimeError("scipy.io.wavfile or noisereduce not available")
    rate, data = wavfile.read(input_file)
    if data.dtype != np.float32:
        data = data.astype("float32") / 32768.0
    reduced = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(output_file, rate, (reduced * 32768).astype("int16"))


# ─── Vietnamese normalization helpers ────────────────────────────────────────
def vietnamese_number_converter(text):
    number_mapping = {
        "không": "0", "hông": "0", "một": "1", "mốt": "1",
        "hai": "2", "ba": "3", "bốn": "4", "năm": "5",
        "sáu": "6", "bảy": "7", "tám": "8", "chín": "9",
    }
    if not text:
        return text
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        normalized = "".join(c for c in words[i].lower() if c.isalpha())
        if normalized in number_mapping:
            seq = []
            punct = ""
            j = i
            while j < len(words) and (
                "".join(c for c in words[j].lower() if c.isalpha()) in number_mapping
            ):
                punct_tmp = "".join(c for c in words[j] if not c.isalpha())
                punct = punct_tmp or punct
                seq.append("".join(c for c in words[j].lower() if c.isalpha()))
                j += 1
            result.append("".join(number_mapping[x] for x in seq) + punct)
            i = j
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)


def convert_vietnamese_diacritics(text):
    char_map = {
        "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
        "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
        "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
        "đ": "d",
        "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
        "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
        "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
        "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
        "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
        "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
        "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
        "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
    }
    uppercase_map = {k.upper(): v.upper() for k, v in char_map.items()}
    char_map.update(uppercase_map)
    return "".join(char_map.get(c, c) for c in text)


def normalize_for_jiwer(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    if unidecode:
        text = unidecode(text)
    else:
        text = convert_vietnamese_diacritics(text)
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


# ─── Match logic ─────────────────────────────────────────────────────────────
def compare_support_dialect_tone(pred: str, expected: str) -> bool:
    """Khớp lệnh: Không phân biệt hoa thường, không dấu, không ký tự đặc biệt."""
    s1 = convert_vietnamese_diacritics(pred.lower())
    s2 = convert_vietnamese_diacritics(expected.lower())
    s1_clean = re.sub(r"[\W_]", "", s1)
    s2_clean = re.sub(r"[\W_]", "", s2)
    return s2_clean in s1_clean


# ─── Transcription ────────────────────────────────────────────────────────────
def transcribe_wav2vec(audio_path, proc, model, device):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = proc(audio, sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    return proc.batch_decode(torch.argmax(logits, dim=-1))[0].strip()


# ─── Checkpoint loading helpers ───────────────────────────────────────────────
def _remap_state_dict_keys(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        remapped[new_key] = value
    return remapped


def _try_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded checkpoint (strict=False).")
        return True
    except Exception:
        remapped_state = _remap_state_dict_keys(state_dict)
        try:
            model.load_state_dict(remapped_state, strict=False)
            logger.info("Loaded checkpoint after remapping (strict=False).")
            return True
        except Exception:
            return False


def _prepare_checkpoint_candidates(ckpt):
    candidates = []
    if not isinstance(ckpt, dict):
        return candidates
    if "state_dict" in ckpt:
        candidates.append(ckpt["state_dict"])
    if "model" in ckpt:
        candidates.append(ckpt["model"])
    candidates.append(ckpt)
    return candidates


def _convert_checkpoint_to_tensor_dict(cand):
    cand_torch = {}
    for key, value in cand.items():
        if isinstance(value, torch.Tensor):
            cand_torch[key] = value
        else:
            try:
                cand_torch[key] = torch.as_tensor(value)
            except Exception:
                pass
    return cand_torch


def try_load_checkpoint_into_model(model, checkpoint_path):
    if checkpoint_path.endswith(".safetensors") and load_safetensors is not None:
        try:
            sd = load_safetensors(checkpoint_path)
            sd = {k: torch.as_tensor(v).cpu() for k, v in sd.items()}
            if "model" in sd and isinstance(sd["model"], dict):
                sd = sd["model"]
            model.load_state_dict(_remap_state_dict_keys(sd), strict=False)
            return True
        except Exception as e:
            logger.warning("Failed reading safetensors: %s", e)
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(_remap_state_dict_keys(sd), strict=False)
        return True
    except Exception:
        return False


def _deserialize_model_loader(model_id):
    model_loader = None
    try:
        model_script = hf_hub_download(repo_id=model_id, filename="model_handling.py")
        model_loader = SourceFileLoader("model_handling", model_script).load_module()
        logger.info("Downloaded model_handling.py from %s", model_id)
    except Exception as e:
        logger.warning(
            "Could not download model_handling.py: %s. Will fallback to AutoModelForCTC.", e
        )
    return model_loader


def _load_processor(model_id, model_dir):
    if model_dir and os.path.isdir(model_dir):
        logger.info("Loading processor from local model_dir: %s", model_dir)
        return Wav2Vec2Processor.from_pretrained(model_dir)
    logger.info("Loading processor from hub model_id: %s", model_id)
    return Wav2Vec2Processor.from_pretrained(model_id)


def _instantiate_model(model_id, model_loader):
    if model_loader is not None and hasattr(model_loader, "Wav2Vec2ForCTC"):
        logger.info("Instantiating custom Wav2Vec2ForCTC from model_handling.py")
        return model_loader.Wav2Vec2ForCTC.from_pretrained(model_id, trust_remote_code=True)
    logger.info("Falling back to AutoModelForCTC.from_pretrained(model_id)")
    from transformers import AutoModelForCTC
    return AutoModelForCTC.from_pretrained(model_id)


def _load_model_and_processor(model_id, model_dir):
    model_loader = _deserialize_model_loader(model_id)
    processor = _load_processor(model_id, model_dir)
    model = _instantiate_model(model_id, model_loader)
    return model, processor


def _load_local_weights(model, local_weights):
    if not local_weights or not os.path.exists(local_weights):
        logger.info("No local_weights provided or not found: %s", local_weights)
        return False
    logger.info("Attempting to load local weights from %s", local_weights)
    try:
        loaded = try_load_checkpoint_into_model(model, local_weights)
    except Exception as e:
        logger.warning("Error while loading local weights: %s", e)
        loaded = False
    if not loaded:
        logger.warning(
            "Could not load local_weights fully. Continuing with hub weights (may be unfine-tuned)."
        )
    return loaded


def _resolve_output_dir(model_dir, local_weights, out_save_dir):
    if out_save_dir:
        return out_save_dir
    if model_dir:
        return model_dir
    if local_weights:
        return os.path.dirname(local_weights)
    return "./out_eval"


# ─── Preprocessing ────────────────────────────────────────────────────────────
def _preprocess_wav(wav, norm_path):
    if AudioSegment is None or wavfile is None or nr is None:
        return wav
    try:
        normalize_audio_pydub(wav, norm_path)
        remove_noise(norm_path, norm_path)
        return norm_path
    except Exception as e:
        logger.warning("Preprocessing failed for %s: %s; using original file.", wav, e)
        return wav


# ─── WER summary ─────────────────────────────────────────────────────────────
def _print_wer_summary(num_pass, num_test, refs, hyps):
    if num_test == 0:
        print("❌ No .wav files found to evaluate.")
        return None, None
    print(f"Total pass: {num_pass}/{num_test} ~ {num_pass * 100 / num_test:.2f}%")
    if wer is None or cer is None:
        logger.warning("jiwer not installed; cannot compute WER/CER.")
        return None, None
    try:
        wer_score = wer(refs, hyps)
        cer_score = cer(refs, hyps)
        print(f"JIwer WER: {wer_score:.4f}")
        print(f"JIwer CER: {cer_score:.4f}")
        return wer_score, cer_score
    except Exception as e:
        print("Could not compute jiwer WER/CER:", e)
        return None, None


# ─── Main evaluation ──────────────────────────────────────────────────────────
def evaluate_folder(  # noqa: C901
    wav_dir,
    model_id=None,
    model_dir=None,
    local_weights=None,
    out_save_dir=None,
    run_postprocess=False,
    run_name=None,
    device=DEVICE_DEFAULT,
    mlflow_tracking_uri=MLFLOW_TRACKING_URI_DEFAULT,
    mlflow_experiment=MLFLOW_EXPERIMENT_DEFAULT,
):
    if not wav_dir or not os.path.exists(wav_dir):
        logger.error("wav_dir not found: %s", wav_dir)
        return

    model_id = model_id or MODEL_ID_DEFAULT
    out_save_dir = _resolve_output_dir(model_dir, local_weights, out_save_dir)
    os.makedirs(out_save_dir, exist_ok=True)

    setup_mlflow(mlflow_tracking_uri, mlflow_experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "wav_dir": wav_dir,
            "model_id": model_id,
            "model_dir": model_dir or "",
            "local_weights": local_weights or "",
            "run_postprocess": run_postprocess,
            "device": device,
            "run_name": run_name or "",
        })

        # DVC provenance
        try:
            import subprocess
            result = subprocess.run(
                ["dvc", "status", "--json"], capture_output=True, text=True, cwd=wav_dir
            )
            if result.returncode == 0:
                mlflow.set_tag("dvc.eval_status", result.stdout.strip()[:500])
        except Exception:
            pass

        model, processor = _load_model_and_processor(model_id, model_dir)
        model.eval()
        _load_local_weights(model, local_weights)
        model.to(device)
        model.eval()

        csv_out = os.path.join(out_save_dir, "transcription_results_wav2vec2.csv")
        with open(csv_out, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["path_wav", "expected_name", "transcription"])
            num_pass = num_test = 0
            refs, hyps = [], []

            for wav in glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True):
                fname = os.path.basename(wav)
                expected = re.sub(r"(?:_\d+)?\.wav$", "", fname)
                expected_for_wer = expected.replace("_", " ")

                tmpdir = "tmp"
                os.makedirs(tmpdir, exist_ok=True)
                norm_path = os.path.join(tmpdir, f"{uuid.uuid4()}_norm.wav")

                denoise_path = _preprocess_wav(wav, norm_path) if run_postprocess else wav

                try:
                    pred = transcribe_wav2vec(denoise_path, processor, model, device)
                except Exception as e:
                    logger.exception("Transcription failed for %s: %s", denoise_path, e)
                    pred = ""

                pred_pp = pred
                writer.writerow([wav, expected, pred_pp])
                csv_file.flush()

                ok = compare_support_dialect_tone(pred_pp, expected)
                status = "PASS" if ok else "FAIL"
                print(f"{status} | File: {fname} | Expected: {expected} | Got: {pred_pp}")

                if ok:
                    num_pass += 1
                num_test += 1

                refs.append(normalize_for_jiwer(expected_for_wer))
                hyps.append(normalize_for_jiwer(pred_pp))

                # Cleanup file tạm
                if os.path.exists(norm_path):
                    os.remove(norm_path)

        print("-" * 30)
        wer_score, cer_score = _print_wer_summary(num_pass, num_test, refs, hyps)

        if num_test > 0:
            pass_rate = num_pass * 100 / num_test
            metrics_dict = {
                "eval.num_files": num_test,
                "eval.num_pass": num_pass,
                "eval.pass_rate_pct": pass_rate,
            }
            if wer_score is not None:
                metrics_dict["eval.wer"] = wer_score
            if cer_score is not None:
                metrics_dict["eval.cer"] = cer_score
            mlflow.log_metrics(metrics_dict)

        try:
            mlflow.log_artifact(csv_out, artifact_path="eval_results")
            logger.info("Logged eval CSV to MLflow.")
        except Exception as e:
            logger.warning("Could not log CSV artifact to MLflow: %s", e)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Eval wav2vec2 with MLflow tracking and DVC data versioning")
    p.add_argument("--wav_dir", type=str, required=True, help="Directory with .wav files (recursive)")
    p.add_argument("--model_id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--model_dir", type=str, default=None)
    p.add_argument("--local_weights", type=str, default=None)
    p.add_argument("--out_save_dir", type=str, default=None)
    p.add_argument("--run_postprocess", action="store_true",
                   help="Normalize audio + Vietnamese number postprocessing")
    p.add_argument("--run_name", type=str, default=None,
                   help="MLflow run name tag")
    p.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    p.add_argument("--mlflow_tracking_uri", type=str,
                   default=os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI_DEFAULT))
    p.add_argument("--mlflow_experiment", type=str,
                   default=os.environ.get("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_DEFAULT))
    args = p.parse_args()

    evaluate_folder(
        wav_dir=args.wav_dir,
        model_id=args.model_id,
        model_dir=args.model_dir,
        local_weights=args.local_weights,
        out_save_dir=args.out_save_dir,
        run_postprocess=args.run_postprocess,
        run_name=args.run_name,
        device=args.device,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
    )
