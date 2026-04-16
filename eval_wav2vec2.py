# file: eval_wav2vec2.py
"""
Evaluation script that loads model architecture from hub (model_id)
AND local safetensors (model.safetensors) to ensure tokenizer <->
weights match the architecture used during finetune.

This version accepts both --model_id and --model_dir (local folder).
If --model_dir is provided it will prefer the local
processor/tokenizer, and it will still download model_handling.py
from --model_id to instantiate the exact model architecture before
loading local weights.

Experiment results are tracked with MLflow; dataset provenance is
managed via DVC.

Usage example:
python eval_wav2vec2.py \
  --wav_dir /content/person_name_500/ \
  --model_dir /content/wav2vec2-finetuned \
  --local_weights /content/wav2vec2-finetuned/model.safetensors \
  --run_postprocess
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

# MLflow defaults
MLFLOW_TRACKING_URI_DEFAULT = "http://mlflow:5000"
MLFLOW_EXPERIMENT_DEFAULT = "wav2vec2-vietnamese-eval"


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


# ------------------ Audio helpers ------------------
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
        raise RuntimeError(
            "scipy.io.wavfile or noisereduce not available"
        )
    rate, data = wavfile.read(input_file)
    if data.dtype != np.float32:
        data = data.astype("float32") / 32768.0
    reduced = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(output_file, rate, (reduced * 32768).astype("int16"))


# ------------------ Vietnamese normalization helpers ------------------
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
                "".join(c for c in words[j].lower() if c.isalpha())
                in number_mapping
            ):
                punct_tmp = "".join(
                    c for c in words[j] if not c.isalpha()
                )
                punct = punct_tmp or punct
                seq.append(
                    "".join(c for c in words[j].lower() if c.isalpha())
                )
                j += 1
            result.append(
                "".join(number_mapping[x] for x in seq) + punct
            )
            i = j
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)


def convert_vietnamese_diacritics(text):
    char_map = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'đ': 'd',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    }
    uppercase_map = {k.upper(): v.upper() for k, v in char_map.items()}
    char_map.update(uppercase_map)
    return "".join(char_map.get(c, c) for c in text)


def convert_vietnamese_number(text: str) -> str:
    char_map = {
        '0': 'không', '1': 'một', '2': 'hai', '3': 'ba',
        '4': 'bốn', '5': 'năm', '6': 'sáu', '7': 'bảy',
        '8': 'tám', '9': 'chín', '10': 'mười',
    }
    return "".join(char_map.get(ch, ch) for ch in text)


def normalize_speech_patterns(text: str) -> str:
    text = text.lower()
    if text.startswith("l"):
        text = "n" + text[1:]
    if text.startswith("r"):
        text = "d" + text[1:]
    if text.startswith("gi"):
        text = "d" + text[2:]
    if text.startswith("s"):
        text = "x" + text[1:]
    if text.startswith("tr"):
        text = "ch" + text[2:]
    return text


def compare_support_dialect_tone(s1: str, s2: str) -> bool:
    s1 = convert_vietnamese_diacritics(s1.lower())
    s2 = convert_vietnamese_diacritics(s2.lower())
    s1_clean = re.sub(r"[\W_]", "", s1)
    s2_clean = re.sub(r"[\W_]", "", s2)
    return s2_clean in s1_clean


def normalize_for_jiwer(text: str) -> str:
    """Chuẩn hóa văn bản để đánh giá WER/CER công bằng nhất."""
    if not text:
        return ""
    text = text.lower()

    # Đưa về tiếng Việt không dấu
    if unidecode:
        text = unidecode(text)
    else:
        text = convert_vietnamese_diacritics(text)

    # Xóa gạch dưới, giữ lại chữ và số
    text = text.replace("_", " ")
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Xóa khoảng trắng thừa
    return " ".join(text.split())


def transcribe_wav2vec(audio_path, processor_ref, model_ref, device):
    audio_arr, sr = librosa.load(audio_path, sr=16000)
    inputs = processor_ref(audio_arr, sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model_ref(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    # Dùng output_word_delimiters=True nếu muốn kiểm soát thủ công,
    # nhưng thông thường cứ để mặc định và strip()
    transcription = processor_ref.batch_decode(pred_ids)[0]
    return transcription.strip()


# ------------------ Checkpoint loading helper ------------------
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
            logger.info(
                "Loaded checkpoint after remapping (strict=False)."
            )
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
    if (
        checkpoint_path.endswith(".safetensors")
        and load_safetensors is not None
    ):
        try:
            sd = load_safetensors(checkpoint_path)
            sd_torch = {
                k: torch.as_tensor(v).cpu() for k, v in sd.items()
            }
            if "model" in sd_torch and isinstance(sd_torch["model"], dict):
                sd_torch = sd_torch["model"]
            if _try_load_state_dict(model, sd_torch):
                return True
        except Exception as e:
            logger.warning("Failed reading safetensors: %s", e)

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return False

    for cand in _prepare_checkpoint_candidates(ckpt):
        cand_torch = _convert_checkpoint_to_tensor_dict(cand)
        if _try_load_state_dict(model, cand_torch):
            return True
    return False


def _deserialize_model_loader(model_id):
    model_loader = None
    try:
        model_script = hf_hub_download(
            repo_id=model_id, filename="model_handling.py"
        )
        model_loader = SourceFileLoader(
            "model_handling", model_script
        ).load_module()
        logger.info("Downloaded model_handling.py from %s", model_id)
    except Exception as e:
        logger.warning(
            "Could not download model_handling.py: %s. "
            "Will fallback to AutoModelForCTC.", e,
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
        logger.info(
            "Instantiating custom Wav2Vec2ForCTC from model_handling.py"
        )
        ModelClass = model_loader.Wav2Vec2ForCTC
        return ModelClass.from_pretrained(model_id, trust_remote_code=True)
    logger.info("Falling back to AutoModelForCTC.from_pretrained(model_id)")
    from transformers import AutoModelForCTC
    return AutoModelForCTC.from_pretrained(model_id)


def _save_model_snapshot(model, processor, out_save_dir):
    os.makedirs(out_save_dir, exist_ok=True)
    try:
        model.save_pretrained(out_save_dir)
    except Exception as e:
        logger.warning(
            "model.save_pretrained() failed: %s; saving state_dict instead.", e,
        )
        torch.save(
            model.state_dict(),
            os.path.join(out_save_dir, "pytorch_model.bin"),
        )
    processor.save_pretrained(out_save_dir)
    logger.info("Saved processor and model state to %s", out_save_dir)


def _load_model_and_processor(model_id, model_dir):
    model_loader = _deserialize_model_loader(model_id)
    processor = _load_processor(model_id, model_dir)
    model = _instantiate_model(model_id, model_loader)
    return model, processor


def _load_local_weights(model, local_weights):
    if not local_weights or not os.path.exists(local_weights):
        logger.info(
            "No local_weights provided or not found: %s", local_weights,
        )
        return False

    logger.info("Attempting to load local weights from %s", local_weights)
    try:
        loaded = try_load_checkpoint_into_model(model, local_weights)
    except Exception as e:
        logger.warning("Error while loading local weights: %s", e)
        loaded = False
    if not loaded:
        logger.warning(
            "Could not load local_weights fully. "
            "Continuing with hub weights (may be unfine-tuned)."
        )
    return loaded


# ------------------ Main evaluation flow ------------------
def _preprocess_wav(wav, norm_path):
    """Normalize and denoise a WAV file; return path to use."""
    if AudioSegment is None or wavfile is None or nr is None:
        return wav
    try:
        normalize_audio_pydub(wav, norm_path)
        remove_noise(norm_path, norm_path)
        return norm_path
    except Exception as e:
        logger.warning(
            "Preprocessing failed for %s: %s; using original file.", wav, e,
        )
        return wav


def _print_wer_summary(num_pass, num_test, refs, hyps):
    """Print pass-rate and optionally jiwer WER & CER."""
    if num_test == 0:
        print("\u274c No .wav files found to evaluate.")
        return None, None

    print(
        f"Total pass: {num_pass}/{num_test} "
        f"~ {num_pass * 100 / num_test:.2f}%"
    )
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


def _resolve_output_dir(model_dir, local_weights, out_save_dir):
    if out_save_dir:
        return out_save_dir
    if model_dir:
        return model_dir
    if local_weights:
        return os.path.dirname(local_weights)
    return "./out_eval"


def evaluate_folder( # noqa: C901
    wav_dir,
    model_id=None,
    model_dir=None,
    local_weights=None,
    out_save_dir=None,
    run_postprocess=False,
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

    # ── MLflow setup ──────────────────────────────────────────────────────────
    setup_mlflow(mlflow_tracking_uri, mlflow_experiment)

    with mlflow.start_run():
        # Log eval configuration
        mlflow.log_params({
            "wav_dir": wav_dir,
            "model_id": model_id,
            "model_dir": model_dir or "",
            "local_weights": local_weights or "",
            "run_postprocess": run_postprocess,
            "device": device,
        })

        # Log DVC data provenance for eval set
        try:
            import subprocess
            result = subprocess.run(
                ["dvc", "status", "--json"],
                capture_output=True, text=True, cwd=wav_dir
            )
            if result.returncode == 0:
                mlflow.set_tag("dvc.eval_status", result.stdout.strip()[:500])
        except Exception:
            pass

        # Nạp Model
        model, processor = _load_model_and_processor(model_id, model_dir)
        model.eval()
        _load_local_weights(model, local_weights)
        model.to(device)
        model.eval()
        _save_model_snapshot(model, processor, out_save_dir)

        # Evaluate WAV files
        csv_out = os.path.join(
            out_save_dir, "transcription_results_wav2vec2.csv"
        )
        with open(csv_out, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["path_wav", "expected_name", "transcription"])
            num_pass = num_test = 0
            refs = []
            hyps = []

            pattern = os.path.join(wav_dir, "**", "*.wav")
            for wav in glob.glob(pattern, recursive=True):
                fname = os.path.basename(wav)

                # 1. Lấy tên gốc từ file
                expected = re.sub(r"(?:_\d+)?\.wav$", "", fname)

                # 2. Xử lý khoảng trắng cho expected
                expected_for_wer = expected.replace("_", " ")

                tmpdir = "tmp"
                os.makedirs(tmpdir, exist_ok=True)
                norm_path = os.path.join(
                    tmpdir, f"{uuid.uuid4()}_norm.wav"
                )
                denoise_path = _preprocess_wav(wav, norm_path)
                try:
                    pred = transcribe_wav2vec(
                        denoise_path, processor, model, device
                    )
                except Exception as e:
                    logger.exception(
                        "Transcription failed for %s: %s", denoise_path, e,
                    )
                    pred = ""

                pred_pp = (
                    vietnamese_number_converter(pred)
                    if run_postprocess
                    else pred
                )
                writer.writerow([wav, expected, pred_pp])
                csv_file.flush()

                # So khớp lấy Pass/Fail
                if unidecode:
                    clean_pred = unidecode(pred_pp.lower())
                    clean_exp = unidecode(expected.lower())
                else:
                    clean_pred = convert_vietnamese_diacritics(pred_pp.lower())
                    clean_exp = convert_vietnamese_diacritics(expected.lower())

                clean_pred = re.sub(r"[\W_]", "", clean_pred)
                clean_exp = re.sub(r"[\W_]", "", clean_exp)

                ok = clean_exp in clean_pred
                status = "PASS" if ok else "FAIL"
                print(
                    f"{status} | File: {fname} | "
                    f"Expected: {expected} | Got: {pred_pp}"
                )

                if ok:
                    num_pass += 1
                num_test += 1

                # 3. Chuẩn hóa chuỗi trước khi đưa vào hàm đếm WER/CER
                norm_ref = normalize_for_jiwer(expected_for_wer)
                norm_hyp = normalize_for_jiwer(pred_pp)

                refs.append(norm_ref)
                hyps.append(norm_hyp)

        print("-" * 30)
        wer_score, cer_score = _print_wer_summary(num_pass, num_test, refs, hyps)

        # ── Log metrics to MLflow ─────────────────────────────────────────────
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

        # Log CSV as MLflow artifact
        try:
            mlflow.log_artifact(csv_out, artifact_path="eval_results")
            logger.info("Logged eval CSV to MLflow.")
        except Exception as e:
            logger.warning("Could not log CSV artifact to MLflow: %s", e)

        # run compare-name logic on saved CSV
        compare_results = compare_csv_and_print_results(csv_out)

        # Log compare-name metrics
        if compare_results:
            mlflow.log_metrics({
                "compare.num_pass": compare_results["num_pass"],
                "compare.num_fail": compare_results["num_fail"],
                "compare.accuracy_pct": compare_results["accuracy_pct"],
            })


def compare_csv_and_print_results(file_csv: str):
    print("\nRunning compare-name logic on:", file_csv)
    num_pass = num_fail = total = 0
    with open(file_csv, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            audio_file = row[0]
            expected_name = row[1]
            model_transcription = row[2]
            result = compare_support_dialect_tone(
                model_transcription, expected_name
            )
            if not result:
                print(
                    f"FAIL | audio_file: {audio_file} "
                    f"expected_name: {expected_name} "
                    f"model_transcription: {model_transcription}"
                )
                num_fail += 1
            else:
                num_pass += 1
            total += 1

    print("\nCompare-name summary:")
    print("Số lượng pass: ", num_pass)
    print("Số lượng fail: ", num_fail)
    accuracy_pct = 0.0
    if total > 0:
        accuracy_pct = num_pass * 100 / total
        print("Tỉ lệ đúng:", f"{accuracy_pct:.2f}%")
    else:
        print("No rows in CSV to compare.")

    return {
        "num_pass": num_pass,
        "num_fail": num_fail,
        "total": total,
        "accuracy_pct": accuracy_pct,
    }


def build_ctcdecoder(labels, kenlm_model_path=None, alpha=0.5, beta=1.5):
    """Build a pyctcdecode beam-search CTC decoder."""
    try:
        from pyctcdecode import build_ctcdecoder as _build
        return _build(
            labels,
            kenlm_model=kenlm_model_path,
            alpha=alpha,
            beta=beta,
        )
    except Exception as e:
        logger.warning("Could not build ctcdecoder: %s", e)
        return None


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Eval wav2vec2 with MLflow tracking and DVC data versioning"
        )
    )
    p.add_argument(
        "--wav_dir", type=str, required=True,
        help="Directory with .wav files (recursive)",
    )
    p.add_argument(
        "--model_id", type=str, default=f"{MODEL_ID_DEFAULT}",
        help=(
            "Hub repo id (used to get model_handling.py and architecture)"
        ),
    )
    p.add_argument(
        "--model_dir", type=str, default=None,
        help=(
            "Local model folder (containing tokenizer/config); "
            "preferred for processor files"
        ),
    )
    p.add_argument(
        "--local_weights", type=str, default=None,
        help="Path to local safetensors or torch checkpoint (optional)",
    )
    p.add_argument(
        "--out_save_dir", type=str, default=None,
        help=(
            "Where to save processor + CSV "
            "(defaults to model_dir or local_weights dir)"
        ),
    )
    p.add_argument(
        "--run_postprocess", action="store_true",
        help="Apply vietnamese number postprocessing",
    )
    p.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    # ── MLflow arguments ──────────────────────────────────────────────────────
    p.add_argument(
        "--mlflow_tracking_uri", type=str,
        default=os.environ.get(
            "MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI_DEFAULT
        ),
        help="MLflow tracking server URI.",
    )
    p.add_argument(
        "--mlflow_experiment", type=str,
        default=os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_DEFAULT
        ),
        help="MLflow experiment name.",
    )
    args = p.parse_args()

    evaluate_folder(
        wav_dir=args.wav_dir,
        model_id=args.model_id,
        model_dir=args.model_dir,
        local_weights=args.local_weights,
        out_save_dir=args.out_save_dir,
        run_postprocess=args.run_postprocess,
        device=args.device,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
    )
