#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QWEN3.5-4B · 交互式激活干预问答

每次只把你敲的那一句话原样喂给模型（不带任何上下文、无系统提示、关 thinking），
并且可以随时选择按怎样的干预强度回答。

用法:
    python interactive_chat.py                          # 默认本地快照 + 0 基层号
    python interactive_chat.py --model <路径或HF id>
    python interactive_chat.py --dose 0.25              # 初始强度

命令（在输入框里直接敲）:
    /dose          查看/修改干预强度（0 = 原模型；可输入任意数，如 0.5、-1）
    /max           查看/修改输出 token 上限（默认 1024）
    /help          显示帮助
    /quit 或 EOF   退出

输出为流式（边生成边打印）；生成过程中按 Ctrl+C 只中断本次回答，不退出程序。
"""

import argparse
import queue
import time
from pathlib import Path
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from activation_intervention import (
    build_prompt,
    calibrate,
    install_hooks,
    resolve_model,
    MODEL_ID,
)

HELP_TEXT = """\
可用命令:
  /dose [强度]   不带参数 = 显示当前强度；带参数 = 设置强度
                 0 = 原模型（不干预），0.25 = 1/4 强度，1 = 完整强度，任意实数均可
  /max [N]       不带参数 = 显示当前输出上限；带参数 = 设置最大输出 token 数
  /help          显示本帮助
  /quit          退出（直接回车/EOF 也退出）
其它任何输入都当作问题，只把这一句发给模型。
"""


def fmt_dose(dose: float) -> str:
    if dose == 0.0:
        return "0（原模型，不干预）"
    return f"{dose:+g}"


class StopFlag(StoppingCriteria):
    """Ctrl+C 置位后，generate 下一个 token 即停。"""

    def __init__(self):
        self.stopped = False

    def __call__(self, input_ids, scores, **kwargs):
        return self.stopped


def stream_generate(model, tokenizer, prompt, max_new, stopper):
    """流式生成：后台线程跑 generate，主线程边收边打印。

    返回 (生成文本, token 数, 是否撞上限, 是否被 Ctrl+C 中断)。
    Ctrl+C 只置停止位并中断本次打印，不抛出到主循环。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.2,
    )
    result = {}

    def _run():
        try:
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new,
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([stopper]),
                )
            gen_ids = out[0][inputs.input_ids.shape[1]:]
            n = gen_ids.shape[0]
            hit = n >= max_new and int(gen_ids[-1]) != tokenizer.eos_token_id
            result.update(text=tokenizer.decode(gen_ids, skip_special_tokens=True),
                          ntok=n, hit=hit)
        except Exception as e:  # noqa: BLE001 — 线程里必须回传，否则主线程死等
            result["error"] = e

    worker = Thread(target=_run, daemon=True)
    worker.start()

    interrupted = False
    while True:
        try:
            piece = next(streamer)
        except queue.Empty:
            # 超时：worker 还活着就继续等，死了说明出错
            if "error" in result:
                raise result["error"]
            if not worker.is_alive() and "text" not in result:
                raise RuntimeError("生成线程意外退出")
            continue
        except StopIteration:
            break
        except KeyboardInterrupt:
            # 只停生成，不退出程序
            interrupted = True
            stopper.stopped = True
            print("\n… [已 Ctrl+C 停止生成]", end="", flush=True)
            continue
        if not interrupted:
            print(piece, end="", flush=True)

    worker.join(timeout=10)
    if "error" in result:
        raise result["error"]
    if interrupted:
        return "", result.get("ntok", 0), False, True
    return (result.get("text", ""), result.get("ntok", 0),
            result.get("hit", False), False)


def ask_dose(current: float) -> float:
    """交互式问一次强度，非法输入保持原值。"""
    raw = input(
        f"  干预强度 [{fmt_dose(current)}]，直接回车保持不变，"
        f"或输入新值 (0 / 0.25 / 1 / 任意): "
    ).strip()
    if not raw:
        return current
    try:
        return float(raw)
    except ValueError:
        print(f"  ! {raw!r} 不是数字，保持 {fmt_dose(current)}")
        return current


def main():
    ap = argparse.ArgumentParser(description="Qwen3.5-4B 交互式激活干预问答")
    ap.add_argument("--model", default=None,
                    help="模型路径或 HF id（默认：本地快照，找不到则用 "
                         f"{MODEL_ID}）")
    ap.add_argument("--base", type=int, choices=[0, 1], default=0,
                    help="层号基准: 0=L14 即 layers[14]（默认）; 1=L14 即 layers[13]")
    ap.add_argument("--dose", type=float, default=0.0,
                    help="初始干预强度，默认 0（原模型）")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="最大输出 token 数，默认 1024（运行时可用 /max 改）")
    args = ap.parse_args()

    model_path, is_local = resolve_model(args.model)

    print(f"[load] {model_path} (bf16)", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=is_local)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", local_files_only=is_local,
    )
    model.eval()
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    state = install_hooks(model, args.base)
    dose = args.dose
    max_new = args.max_new_tokens
    state["dose"] = dose

    print()
    print("=" * 60)
    print(" 交互式激活干预问答 · 一句话一答（无上下文）")
    print(f" 当前强度: {fmt_dose(dose)}   输出上限: {max_new} tok"
          f"   /help 看命令   /quit 退出")
    print("=" * 60)

    while True:
        try:
            print()
            q = input(f"[强度 {fmt_dose(dose)}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见～")
            break
        if not q:
            continue
        if q in ("/quit", "/exit", "/q"):
            break
        if q == "/help":
            print(HELP_TEXT)
            continue
        if q == "/dose" or q.startswith("/dose "):
            parts = q.split(maxsplit=1)
            if len(parts) == 1:
                print(f"  当前强度: {fmt_dose(dose)}")
            else:
                try:
                    dose = float(parts[1])
                    print(f"  强度已设为 {fmt_dose(dose)}")
                except ValueError:
                    print(f"  ! {parts[1]!r} 不是数字")
            continue

        if q == "/max" or q.startswith("/max "):
            parts = q.split(maxsplit=1)
            if len(parts) == 1:
                print(f"  当前输出上限: {max_new} tok")
            else:
                try:
                    n = int(parts[1])
                    if n <= 0:
                        raise ValueError
                    max_new = n
                    print(f"  输出上限已设为 {max_new} tok")
                except ValueError:
                    print(f"  ! {parts[1]!r} 不是正整数")
            continue

        # 标定必须在卸力状态（dose=0）下跑，否则量到的残差流被干预污染，
        # scale 会滚雪球式膨胀（曾炸到 1e8）。
        state["dose"] = 0.0
        prompt = build_prompt(tokenizer, q)
        state["scale"] = calibrate(model, tokenizer, prompt, args.base)
        state["dose"] = dose
        stopper = StopFlag()
        t = time.time()
        print()
        text, ntok, hit, interrupted = stream_generate(
            model, tokenizer, prompt, max_new, stopper)
        state["dose"] = 0.0
        pretty = ", ".join(f"L{L}N{n}={v:.4g}"
                           for (L, n), v in state["scale"].items())
        if interrupted:
            flag = "  [已中断]"
        elif hit:
            flag = "  [达 token 上限]"
        else:
            flag = ""
        print(f"\n--- {ntok} tok / {time.time() - t:.2f}s{flag} "
              f"[标定 {pretty}] ---")

    model.cpu()
    torch.cuda.empty_cache()
    print("[bye]")


if __name__ == "__main__":
    main()
