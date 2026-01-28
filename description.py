# -*- coding: utf-8 -*-
"""
generate_vlm_exp5.py

Use Qwen3-VL-Plus to generate VLM text augmentation for Experiment 5.

Assumptions:
- data/{guid}.jpg exists
- guid is integer from 1 to 5129 (continuous)
- output: data/{guid}.vlm.txt
"""

import os
import time
import json
import base64
import argparse
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# Config
# ============================================================
PROMPT = (
    "You are a multimodal meme-understanding assistant.\n"
    "Please analyze this meme and produce a concise English description with THREE parts:\n"
    "1) OCR: list ALL readable texts on the image (verbatim if possible).\n"
    "2) Visual: describe key visual elements (people/objects/background/style).\n"
    "3) Meaning: combine text+visual to explain the intended stance, tone "
    "(e.g., satire/irony), and sentiment.\n"
    "Keep it compact (120–180 words). Do NOT include class labels.\n"
)

# ============================================================
# Utils
# ============================================================
def img_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def call_qwen_vl(
    client: OpenAI,
    model: str,
    image_path: str,
    enable_thinking: bool,
    thinking_budget: int,
    timeout: int,
):
    img_url = img_to_data_url(image_path)

    extra_body = (
        {"enable_thinking": True, "thinking_budget": thinking_budget}
        if enable_thinking
        else None
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }
        ],
        stream=False,
        timeout=timeout,
        extra_body=extra_body,
    )

    return (resp.choices[0].message.content or "").strip()


# ============================================================
# Main
# ============================================================
def main(args):
    client = OpenAI(
        api_key="sk-9d3da10b79484c24bc52922d3193a60f",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    os.makedirs(args.data_dir, exist_ok=True)

    # ----------------------------
    # Cache (for resume)
    # ----------------------------
    cache = {}
    if os.path.exists(args.cache):
        try:
            with open(args.cache, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    print("=" * 60)
    print("Experiment 5 VLM Augmentation")
    print(f"Data dir      : {args.data_dir}")
    print(f"GUID range    : 1 ~ {args.max_guid}")
    print(f"Model         : {args.model}")
    print("=" * 60)

    # ----------------------------
    # Main loop
    # ----------------------------
    for guid in tqdm(range(1, args.max_guid + 1), desc="Generating VLM text"):
        guid_str = str(guid)

        img_path = os.path.join(args.data_dir, f"{guid_str}.jpg")
        out_path = os.path.join(args.data_dir, f"{guid_str}.vlm.txt")

        # Image missing (should not happen, but safe)
        if not os.path.exists(img_path):
            continue

        # Already generated
        if os.path.exists(out_path):
            continue

        # Cache hit
        if guid_str in cache and cache[guid_str].strip():
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cache[guid_str])
            continue

        # Call VLM (with retry)
        desc = ""
        last_err = None
        for attempt in range(args.max_retries):
            try:
                desc = call_qwen_vl(
                    client=client,
                    model=args.model,
                    image_path=img_path,
                    enable_thinking=args.enable_thinking,
                    thinking_budget=args.thinking_budget,
                    timeout=args.timeout,
                )
                if desc:
                    break
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))

        if not desc:
            desc = f"VLM generation failed: {last_err}"

        # Save
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(desc)

        cache[guid_str] = desc
        with open(args.cache, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    print("\n✅ All done.")
    print("You can now safely use *.vlm.txt in datasets.py")


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("VLM augmentation for Experiment 5")

    parser.add_argument("--data_dir", default="./data", help="Directory containing jpg files")
    parser.add_argument("--max_guid", type=int, default=5129, help="Max guid index")
    parser.add_argument("--model", default="qwen3-vl-plus", help="Qwen VLM model")
    parser.add_argument("--api_key", required=False, help="DashScope API key")

    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--thinking_budget", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)

    parser.add_argument("--cache", default="./vlm_cache_exp5.json")

    args = parser.parse_args()
    main(args)
 