from __future__ import annotations

from collections.abc import Iterable


class ParaphraseGenerator:
    def __init__(
        self,
        model_name: str = "eugenesiow/bart-paraphrase",
        allow_download: bool = False,
        max_new_tokens: int = 96,
    ) -> None:
        self.model_name = model_name
        self.allow_download = allow_download
        self.max_new_tokens = max_new_tokens
        self._bundle = None

    def _load_bundle(self):
        if self._bundle is not None:
            return self._bundle
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            self._bundle = False
            return self._bundle
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True, use_fast=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
        except Exception:
            if not self.allow_download:
                self._bundle = False
                return self._bundle
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.eval()
        self._bundle = (tokenizer, model)
        return self._bundle

    def generate(self, task: str, response: str, variants: int = 3) -> tuple[tuple[str, str], ...]:
        bundle = self._load_bundle()
        if bundle is False:
            return tuple()
        tokenizer, model = bundle
        prompts = (
            ("paraphrase_semantic", f"paraphrase: {response}"),
            ("paraphrase_authoritative", f"paraphrase with formal authoritative wording: {response}"),
            ("paraphrase_structured", f"paraphrase with structured concise wording: {response}"),
            ("paraphrase_confident", f"paraphrase with confident wording: {response}"),
        )
        outputs: list[tuple[str, str]] = []
        for label, prompt in prompts[:variants]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=0.92,
                temperature=0.8,
                num_return_sequences=1,
            )
            text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            if text and text != response:
                outputs.append((label, text))
        deduped: list[tuple[str, str]] = []
        seen = set()
        for label, text in outputs:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append((label, text))
        return tuple(deduped)


def merge_generated_candidates(
    generated: Iterable[tuple[str, str]],
    existing_ops: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((text, existing_ops + (label,)) for label, text in generated)
