#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
剂量扫描：找出真正能撬动行为的干预量级。

对每个问题跑贪心生成，与 baseline（dose=0）逐 token 比较：
报告第一个分叉 token 的位置，以及最终文本是否实质不同。
同时扫两种层号基准（0 基 / 1 基）。
"""

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_intervention import (
    QUESTIONS,
    build_prompt,
    install_hooks,
    resolve_model,
)

DOSES = [1, 10, 50, 200]
BASES = [0, 1]
QS = [QUESTIONS[0], QUESTIONS[5]]  # 道德题 + 数学题
MAX_NEW = 180


@torch.inference_mode()
def gen_ids(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW,
                         pad_token_id=tokenizer.eos_token_id)
    return out[0][inputs.input_ids.shape[1]:]


def main():
    model_path, is_local = resolve_model()
    print(f"[load] {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=is_local)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", local_files_only=is_local,
    )
    model.eval()

    prompts = [build_prompt(tok, q) for q in QS]

    # install_hooks 不支持卸载，所以逐 base 串行：每个 base 重新加载模型，
    # 装钩子 → 跑本 base 的 baseline（dose=0 是 no-op）→ 扫剂量。
    print("\n" + "=" * 78)
    print(f"{'问题':<6} {'基准':>4} {'dose':>6} | {'首分叉':>8} {'长度':>5} {'实质不同':>8}")
    print("-" * 78)

    # 由于 install_hooks 无法卸载，重新加载模型逐 base 干净地跑
    for base in BASES:
        # 重新加载以确保没有残留 hook
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa", local_files_only=is_local,
        )
        model.eval()
        state = install_hooks(model, base)
        base_ids = [gen_ids(model, tok, p) for p in prompts]  # 本 base 的 baseline
        state["dose"] = 0.0

        for qi, p in enumerate(prompts):
            ref = base_ids[qi]
            for dose in DOSES:
                state["dose"] = float(dose)
                ids = gen_ids(model, tok, p)
                state["dose"] = 0.0
                # 首分叉
                div = None
                for k in range(min(len(ref), len(ids))):
                    if int(ref[k]) != int(ids[k]):
                        div = k
                        break
                if div is None and len(ref) != len(ids):
                    div = min(len(ref), len(ids))
                same = torch.equal(ref, ids)
                txt = tok.decode(ids, skip_special_tokens=True)
                head = txt[:34].replace("\n", " ")
                print(f"Q{qi + 1}{'':<4} {base:>4} {dose:>6g} | "
                      f"{'—' if div is None else f'#{div}':>8} {len(ids):>5} "
                      f"{'否' if same else '是':>8}  {head}", flush=True)

    print("=" * 78)


if __name__ == "__main__":
    main()
