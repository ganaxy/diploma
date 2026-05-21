"""Input x Gradient token-importance explainer.

For a given (model, text, predicted-class) triple, returns one (token, score)
tuple per sub-word so the UI can render it through gr.HighlightedText.

Method (lightweight, no Captum):
  1. Tokenize and look up input embeddings.
  2. Mark embeddings as requires_grad and run the forward pass via inputs_embeds.
  3. Take the logit of the predicted class and backward.
  4. attribution[t] = sum_d ( embedding[t,d] * grad[t,d] ).
  5. Mask special tokens, take absolute value, normalise to [0, 1].

The output preserves the SentencePiece sub-word boundaries (the model uses an
AlbertTokenizer with a sentencepiece "spiece.model"). Leading "_" markers are
mapped back to a leading space so the highlighted output reads naturally.
"""

from typing import List, Tuple

import numpy as np
import torch


def _readable_token(tok: str) -> str:
    """SentencePiece pieces start with U+2581 ('_') to mark a word boundary."""
    if tok.startswith("▁"):
        return " " + tok[1:]
    return tok


def token_attributions(
    single_model,            # gradio_app.models._SingleModel
    text: str,
    predicted_label: str,
    max_length: int = None,  # ignored; the model's own max_length is used
) -> List[Tuple[str, float]]:
    """Return [(displayed_token, normalised_importance), ...] for one example.

    Attribution is computed on the SAME training-faithful input the model was
    scored on (source prefix + normalisation for Flat), so the highlighted
    tokens correspond to what actually drove the prediction.
    """
    tokenizer = single_model.tokenizer
    model = single_model.model
    device = single_model.device
    model_input = single_model.build_input(text)

    enc = tokenizer(
        model_input,
        padding=True,
        truncation=True,
        max_length=single_model.max_length,
        return_tensors="pt",
    ).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).clone().detach()
    inputs_embeds.requires_grad_(True)

    # Toggle eval mode but allow gradients through the embedding tensor.
    was_training = model.training
    model.eval()
    try:
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        target_idx = single_model.model.config.label2id[predicted_label]
        target_logit = outputs.logits[0, target_idx]
        model.zero_grad(set_to_none=True)
        target_logit.backward()
    finally:
        if was_training:
            model.train()

    grads = inputs_embeds.grad
    if grads is None:
        return [(text, 0.0)]

    # Input x Gradient, summed over the hidden dimension.
    attributions = (inputs_embeds * grads).sum(dim=-1).detach().cpu().numpy()[0]
    mask_np = attention_mask[0].detach().cpu().numpy().astype(bool)
    ids = input_ids[0].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids.tolist())

    special_ids = set(tokenizer.all_special_ids)

    keep_idx = [
        i for i in range(len(tokens))
        if mask_np[i] and int(ids[i]) not in special_ids
    ]
    if not keep_idx:
        return [(text, 0.0)]

    raw = np.abs(attributions[keep_idx])
    if raw.max() > 0:
        norm = raw / raw.max()
    else:
        norm = raw

    return [(_readable_token(tokens[i]), float(s)) for i, s in zip(keep_idx, norm)]
