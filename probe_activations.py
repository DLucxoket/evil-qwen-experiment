#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测干预点的真实激活量级：layers[14].mlp.down_proj 输入 N8949、layers[13] N7956。"""

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_intervention import (
    QUESTIONS,
    build_prompt,
    find_mlp_down,
    resolve_model,
)

PROBES = [(14, 8949), (13, 7956)]  # (layer, neuron)


def main():
    model_path, is_local = resolve_model()
    print(f"[load] {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=is_local)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", local_files_only=is_local,
    )
    model.eval()

    captured = {L: [] for L, _ in PROBES}
    handles = []
    for layer, neuron in PROBES:
        mod = find_mlp_down(model, layer)

        def make_hook(layer):
            def hook(module, args):
                x = args[0]
                captured[layer].append(x.detach().float())
            return hook

        handles.append(mod.register_forward_pre_hook(make_hook(layer)))

    for q in QUESTIONS[:4]:
        prompt = build_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=8,
                           do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)

    print(f"\n{'层':>4} {'神经元':>6} | {'该神经元均值':>12} {'标准差':>10} "
          f"{'|值|max':>10} | {'整层均值':>10} {'整层std':>10} {'|层|max':>10}")
    print("-" * 90)
    for layer, neuron in PROBES:
        xs = torch.cat([x.reshape(-1, x.shape[-1]) for x in captured[layer]])
        n = xs[:, neuron]
        print(f"L{layer:<3} N{neuron:<5} | {n.mean():>12.3f} {n.std():>10.3f} "
              f"{n.abs().max():>10.3f} | {xs.mean():>10.3f} {xs.std():>10.3f} "
              f"{xs.abs().max():>10.3f}")
        # 顺带看看该神经元绝对值排全层第几
        rank = (xs.abs().mean(0) > xs[:, neuron].abs().mean()).sum().item()
        print(f"      → 平均绝对值在 {xs.shape[-1]} 维中排第 {rank + 1} 名")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
