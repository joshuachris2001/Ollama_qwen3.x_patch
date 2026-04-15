# 繋
### Project GGUF patcher for Ollama Multimodal Monoliths
This project will take compatible models' pure llama.cpp text and mmproj models and convert them to an Ollama GGUF Monolith model.

This project is modular, instead of its predecessor of spaghetti code 🍝; it is now easier to add model types and multimodal functions. 🧩

### What is a Monolith model?
Ollama's new engine handles GGUF models differently; these models are taken from Hugging Face and converted to a **merged** GGUF format. This reduces memory overhead by storing the multimodal tensors in one place.
This is why pure GGUF models crash Ollama when accompanied with an mmproj; they are formatted wrong or use the wrong tensor formatting.

> ⚠️ **One-way conversion:** The output of this tool is an Ollama-native GGUF monolith. This format is not compatible with llama.cpp or any other llama.cpp-based inference engine (LM Studio, Jan, etc.) and will cause them to crash. This is by design — the merged file contains tensors that only Ollama's runtime understands.

### Origins
* Qwen3.x GGUF vision patcher; This was a simple project made to quickly patch the slightly buggy Qwen3-VL and Qwen3.5 model family in Ollama. Made for people who would prefer Ollama over llama.cpp [*I do wish that Ollama fixes the template handling, but what can we do~*]. [*and little RAM use due to mmap-ing*]

Sparked from the need to find other ways to create Ollama GGUF multimodal models where limited memory is involved; (Will work on lessening the reliance of the source BLOBs over time); default `ollama create` typically requires the entire model loaded into RAM and can cause OOM kills; my system could not handle this, so I had to come up with a quick alternative. GGUF is used here to reroute tensors to expectations [work on quant matching later], these tools use mmapping which reduces the RAM draw considerably.

R&D of brute forcing [Jan-v2-VL](https://ollama.com/fredrezones55/Jan-v2-VL) to work with Ollama, I found the conflicts where Ollama does not like and created a patcher that merges the GGUF models to how Ollama expects. Ollama still has a few kinks, but that is part of the Ollama limitations with how the chat templates are handled.

### What's this about a model BLOB?
To ensure the program had proper tensor configurations, I found it was easier to take vital mmproj and other tensor information from the official vision tensors than hardcoding it — mainly to verify the Ollama vision limits, attention structure, and RoPE information, etc.

You can get the needed model BLOB by downloading the model size for that particular base model;

**This is important:** _For most cases the model of the finetuned model you have needs to be the same as the Ollama GGUF BLOB._

## Preparation

### Recommended: Install via PyPI

The easiest way to get started. Dependencies are managed automatically — no manual installs needed.

**Run directly without installing (uvx):**
```bash
uvx tsunagi-ollama-bridge
```

**Install as a persistent command:**
```bash
uv tool install tsunagi-ollama-bridge
tsuangi-ollama --help
```

**Classic pip install:**
```bash
pip install tsunagi-ollama-bridge
```

### Manual / Development Install

If you are running from source, install dependencies manually:
```bash
pip install gguf tqdm numpy
```

If we are merging a Qwen3.5 model, there are parts of the code that were not hardcoded to get it to work, so we need to download the model your finetune was based off of.

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
- `gemma4` — source blob is required 

`--llm` — the finetuned text model you want to use.

`--mmproj` — the mmproj vision file to merge in with the text model.

`--blob` — the source model blob the Qwen3.5 path needs, as there may be differences between 4B and 27B.

`--output` — where the output merged GGUF model will go. *(Note: by merging the model file, it will no longer be supported by llama.cpp)*

### Examples

**Qwen3-VL finetune (no blob needed):**
```bash
tsuangi-ollama \
    --model-type qwen3vl \
    --llm    my-finetune.Q5_K_M.gguf \
    --mmproj mmproj.gguf \
    --output merged_model.gguf
```

**Qwen3-VL-MOE finetune (no blob needed):**
```bash
tsuangi-ollama \
    --model-type qwen3vlmoe \
    --llm    my-finetune.Q4_K_M.gguf \
    --mmproj mmproj.gguf \
    --output merged_model.gguf
```

**Qwen3.5 finetune (blob required):**
```bash
tsuangi-ollama \
    --model-type qwen35 \
    --blob   /var/lib/ollama/blobs/sha256-81fb60... \
    --llm    my-finetune.Q6_K.gguf \
    --mmproj mmproj.gguf \
    --output merged_qwen35.gguf
```

If `--output` is omitted, the merged model will be saved as `merged.gguf` in the current working directory.

# AI Receipt
I used Claude Sonnet 4.6 (via Perplexity) to assist with the program structure.
