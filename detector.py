from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer


class DetectorError(RuntimeError):
    """Raised when the detector cannot be initialized or run."""


@dataclass
class SlopDetector:
    model_id: str = "gouwsxander/slop-detector-bert"
    max_length: int = 512
    stride: int = 64
    batch_size: int = 4

    def __post_init__(self) -> None:
        self.device = self._select_device()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # AutoPeftModel reads the adapter configuration and loads its base model.
            self.model = AutoPeftModelForSequenceClassification.from_pretrained(
                self.model_id
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise DetectorError(
                f"Could not load '{self.model_id}'. Confirm that you have internet "
                "access and that the model repository is available. If the repository "
                "requires access approval, sign in with `huggingface-cli login`. "
                f"Original error: {exc}"
            ) from exc

        # The model card identifies LABEL_1 as the AI-generated class.
        self.ai_label_id = 1
        if getattr(self.model.config, "num_labels", 2) < 2:
            raise DetectorError("Expected a two-class sequence-classification model.")

        self.max_length = min(
            self.max_length,
            int(getattr(self.tokenizer, "model_max_length", 512))
            if getattr(self.tokenizer, "model_max_length", 512) < 100_000
            else 512,
        )

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @torch.inference_mode()
    def predict(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise DetectorError("Please provide non-empty text.")

        try:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                stride=min(self.stride, self.max_length // 4),
                return_overflowing_tokens=True,
                return_attention_mask=True,
                padding=True,
                return_tensors="pt",
            )
        except Exception as exc:
            raise DetectorError(f"Text tokenization failed: {exc}") from exc

        # Overflow metadata is for bookkeeping, not a model input.
        model_inputs = {}
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in encoded:
                model_inputs[key] = encoded[key]

        input_ids = model_inputs["input_ids"]
        chunk_count = input_ids.shape[0]
        token_count = len(self.tokenizer.encode(text, add_special_tokens=False))

        probabilities: list[float] = []
        chunks: list[dict[str, Any]] = []

        for start in range(0, chunk_count, self.batch_size):
            batch = {
                key: value[start : start + self.batch_size].to(self.device)
                for key, value in model_inputs.items()
            }
            outputs = self.model(**batch)
            batch_probs = torch.softmax(outputs.logits.float(), dim=-1)[:, self.ai_label_id]
            probabilities.extend(batch_probs.detach().cpu().tolist())

        for index, probability in enumerate(probabilities):
            ids = input_ids[index].tolist()
            # Remove padding and special tokens for a readable preview.
            if "attention_mask" in model_inputs:
                mask = model_inputs["attention_mask"][index].tolist()
                ids = [token_id for token_id, keep in zip(ids, mask) if keep]
            chunk_text = self.tokenizer.decode(ids, skip_special_tokens=True)
            chunks.append(
                {
                    "chunk": index + 1,
                    "tokens": len(self.tokenizer.encode(chunk_text, add_special_tokens=False)),
                    "ai_probability": float(probability),
                    "text": chunk_text,
                }
            )

        # Mean across overlapping windows. Treat this as a heuristic score, not a
        # calibrated document-level probability.
        ai_probability = float(sum(probabilities) / len(probabilities))
        predicted_label = "AI-generated" if ai_probability >= 0.5 else "Human-written"

        return {
            "predicted_label": predicted_label,
            "ai_probability": ai_probability,
            "chunk_count": chunk_count,
            "token_count": token_count,
            "device": str(self.device),
            "chunks": chunks,
        }
