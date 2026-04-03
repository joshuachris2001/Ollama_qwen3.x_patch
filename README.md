# Qwen3.x GGUF vision patcher
This is a simple project made to quickly patch the slightly buggy Qwen3-VL and Qwen3.5 model family in Ollama. Made for people who would prefer Ollama over llama.cpp [*I do wish that Ollama fixes the template handling, but what can we do~*].

With this project you can take a plain llama.cpp GGUF model into a direct Ollama-compatible model, no more conversion work necessary!

[*and little RAM use due to mmap-ing*]

### Why?
Sparked from the R&D of brute forcing [Jan-v2-VL](https://ollama.com/fredrezones55/Jan-v2-VL) to work with Ollama, I found the conflicts where Ollama does not like and created a patcher that simply merges the GGUF models to how Ollama expects. Ollama still has a few kinks, but that is part of the Ollama limitations with how the chat templates are handled; after this work I did not want to hog it — and Ollama's built-in `create` requires a LOT of RAM, whereas llama.cpp's tools use mmap-ing [much nicer on memory]. *Is it not a pain that `ollama create` just kills itself due to OOM kills? Why should we suffer with this when llama.cpp tools are more efficient and produce practically the same file formats?*

### What's this about a model BLOB?
To ensure the program had proper tensor configurations, I found it was easier to take vital mmproj information from the official vision tensors than hardcoding it — mainly to verify the Ollama vision limits, attention structure, and RoPE information.

Not completely required for Qwen3-VL models as most of it was hardcoded from taking samples from the blob models directly and pulling these values out (the said brute-forcing to get it to work). It seemed that between the different model sizes with Qwen3.5, hardcoding would not completely cut it.

You can get the model BLOB by downloading the qwen3.5 model size for the base model and looking at the resulting modelfile.

### What's in `advanced/` and `prototype/`?
- **`advanced/`** — contains a reference shell script (`qwen-3-VL-Ollama_tuned_vission_quant.sh`) that documents approximately how the Ollama team quantizes the Qwen3-VL vision encoder. It uses `llama-quantize` with per-layer `--tensor-type` overrides, applying a mixed Q8_0/Q5_0 scheme to each `attn_v` block across all 27 vision layers, then quantizes the rest as F16. Useful if you want your merged mmproj to match Ollama's official quantization quality as closely as possible.
- **`prototype/`** — the original per-architecture scripts (`Qwen3-VL-merge.py`, `Qwen3-VL-MOE-merge.py`, `Qwen3.5-merge.py`) that the unified program evolved from. Useful as a reference for understanding the R&D lineage.

## Preparation
In your Python environment you should have `gguf` installed. I also included `tqdm` for progress bars; we also need `numpy` to handle the arrays.
```
pip install gguf tqdm numpy
```

If we are merging a qwen3.5 model, there are parts of the code that were not hardcoded to get it to work, so we need to download the model your finetune was based off of.

For example Qwen3.5 4B:

```bash
ollama pull qwen3.5:4b
ollama show --modelfile qwen3.5:4b
```

The second `FROM` line will show the full path of the model BLOB on your system.

## Usage
The program has 3–4 important arguments.

`--model-type` — The program needs to be told what model architecture this is (auto-discovery is not yet implemented):
- `qwen3vl` — source blob not required
- `qwen3vlmoe` — source blob not required
- `qwen35` — source blob is required

`--llm` — the finetuned text model you want to use.

`--mmproj` — the mmproj vision file to merge in with the text model.

`--blob` — the source model blob the qwen3.5 path needs, as there may be differences between 4B and 27B.

`--output` — where the output merged GGUF model will go. *(Note: by merging the model file, it will no longer be supported by llama.cpp)*

### Examples

**Qwen3-VL finetune (no blob needed):**
```bash
python qwen-vl-merge-unified.py \
    --model-type qwen3vl \
    --llm    my-finetune.Q5_K_M.gguf \
    --mmproj mmproj.gguf \
    --output merged_model.gguf
```

**Qwen3-VL-MOE finetune (no blob needed):**
```bash
python qwen-vl-merge-unified.py \
    --model-type qwen3vlmoe \
    --llm    my-finetune.Q4_K_M.gguf \
    --mmproj mmproj.gguf \
    --output merged_model.gguf
```

**Qwen3.5 finetune (blob required):**
```bash
python qwen-vl-merge-unified.py \
    --model-type qwen35 \
    --blob   /var/lib/ollama/blobs/sha256-81fb60... \
    --llm    my-finetune.Q6_K.gguf \
    --mmproj mmproj.gguf \
    --output merged_qwen35.gguf
```

If `--output` is omitted, the merged model will be saved as `merged_qwen.gguf` in the current working directory.

# AI Receipt
I used Claude Sonnet 4.6 (via Perplexity) to assist with the program structure and the merging of the merge programs.
