from typing import List, Tuple

import numpy as np
import torch

def _readable_token(tok: str) -> str:
    if tok.startswith("▁"):
        return " " + tok[1:]
    return tok

def token_attributions(
    text: str,
    predicted_label: str,
    max_length: int = 128,
) -> List[Tuple[str, float]]:
    tokenizer = single_model.tokenizer
    model = single_model.model
    device = single_model.device

    enc = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).clone().detach()
    inputs_embeds.requires_grad_(True)

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
