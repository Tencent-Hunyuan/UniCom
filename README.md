# UniCom: Unified Multimodal Modeling via Compressed Continuous Semantic Representations

Official code for the paper [UniCom: Unified Multimodal Modeling via Compressed Continuous Semantic Representations](https://arxiv.org/abs/2603.10702).

[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-UniCom-yellow)](https://huggingface.co/tencent/Unicom-Unified-Multimodal-Modeling-via-Compressed-Continuous-Semantic-Representations)
[![arXiv](https://img.shields.io/badge/arXiv-2603.10702-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.10702)
[![Project Page](https://img.shields.io/badge/💻_Project-UniCom-blue.svg)](https://miazhao7708.github.io/UniComPage/)
[![GitHub](https://img.shields.io/badge/GitHub-UniCom-181717?logo=github&logoColor=white)](https://github.com/Tencent-Hunyuan/UniCom)

UniCom is a unified large-scale multimodal model that performs generation directly over compressed visual embeddings. This repository includes the inference pipeline for text-to-image generation, image editing, and image reconstruction.

![UniCom Framework](./UniCom/assets/framework.png)

*Figure: We compare different unified modeling choices in terms of convergence speed and consistency on editing tasks, and ultimately build UniCom with the Path I transfusion-style formulation rather than the Path II query-guided design.*

## 🔥 Key Contributions

- `Model`: We propose **UniCom**, a unified large-scale multimodal model that performs generation directly over compressed visual embeddings and serves as a unified interface for both understanding and generation.
- `Paradigm`: We establish an effective paradigm for unifying visual understanding and generation by predicting continuous compressed visual embeddings, and show that compressing visual features along the **channel dimension** is a particularly effective way to preserve both semantics and fine-grained details.
- `Results`: UniCom achieves state-of-the-art or competitive performance across image reconstruction, text-to-image generation, and challenging image editing tasks, with especially strong performance on editing benchmarks.

## Setup

### 1. Download Checkpoints

Download all checkpoints at once via `huggingface-cli`:

```bash
huggingface-cli download tencent/Unicom-Unified-Multimodal-Modeling-via-Compressed-Continuous-Semantic-Representations --repo-type model --local-dir ./model_zoo/ --resume-download
```


You can also download each component separately:

| Component | Local Path | Link |
| --- | --- | --- |
| UniCom (text → SigLIP) | `model_zoo/unicom_hf_model/` | [Download](https://huggingface.co/tencent/Unicom-Unified-Multimodal-Modeling-via-Compressed-Continuous-Semantic-Representations/tree/main/unicom_hf_model) |
| Decoder Transformer (SigLIP → image) | `model_zoo/unicom_decoder_transformer.pt` | [Download](https://huggingface.co/tencent/Unicom-Unified-Multimodal-Modeling-via-Compressed-Continuous-Semantic-Representations/blob/main/unicom_decoder_transformer.pt) |
| Flux VAE | `model_zoo/flux-vae/` | [Download](https://huggingface.co/tencent/Unicom-Unified-Multimodal-Modeling-via-Compressed-Continuous-Semantic-Representations/tree/main/flux-vae) |
| SigLIP2| `model_zoo/siglip2-so400m-patch16-naflex/` | [Download](https://huggingface.co/tencent/Unicom-Unified-Multimodal-Modeling-via-Compressed-Continuous-Semantic-Representations/tree/main/siglip2-so400m-patch16-naflex) |

After downloading, verify the expected directory layout:

```text
model_zoo/
├── unicom_hf_model/
├── unicom_decoder_transformer.pt
├── flux-vae/
└── siglip2-so400m-patch16-naflex/
```

### 2. Environment Setup

```bash
conda create -n unicom python=3.12 -y
conda activate unicom
```

Install PyTorch first according to your CUDA version. Example for CUDA 12.8:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -r requirements.txt
```

## 🚀 Usage


### Case 1: Text-to-image generation

```bash
python run_unicom_decoder_pipeline.py \
  --model-path ./model_zoo/unicom_hf_model \
  --prompt "A ginger kitten tangled in a ball of wool, looking puzzled." \
  --output-dir ./output/t2i_demo \
  --diff-infer-steps 50 \
  --seed 42 \
  --image-size auto \
  --n-samples-per-prompt 4
```

### Case 2: Single-image editing

```bash
python run_unicom_decoder_pipeline.py \
  --model-path ./model_zoo/unicom_hf_model \
  --prompt "Add a blue baseball cap on the boy's head" \
  --image ./UniCom/assets/demo_imgs/input_0.jpg \
  --image-size auto \
  --seed 42 \
  --output-dir ./output/ti2i_demo \
  --diff-infer-steps 50
```

### Case 3: Multi-image editing

```bash
python run_unicom_decoder_pipeline.py \
  --model-path ./model_zoo/unicom_hf_model \
  --prompt "Place the chair from the second image onto the snow in the third image, and then place the coffee cup from the first image onto the chair." \
  --image ./UniCom/assets/demo_imgs/input_1_0.png ./UniCom/assets/demo_imgs/input_1_1.png ./UniCom/assets/demo_imgs/input_1_2.png \
  --image-size auto \
  --seed 42 \
  --output-dir ./output/ti2i_multi_demo \
  --diff-infer-steps 50
```

### Case 4: CSV-based batch inference

```bash
python run_unicom_decoder_pipeline.py \
  --model-path ./model_zoo/unicom_hf_model \
  --csv-path ./UniCom/eval/t2i.csv \
  --output-dir ./output/t2i_demo_csv \
  --num-gpus 8 \
  --decoder-device 0,1,2,3,4,5,6,7 \
  --image-size auto \
  --diff-infer-steps 50 \
  --n-samples-per-prompt 4
```


```bash
# no cot
python run_unicom_decoder_pipeline.py \
  --model-path ./model_zoo/unicom_hf_model \
  --csv-path ./UniCom/eval/t2i.csv \
  --output-dir ./output/t2i_demo_csv_nocot \
  --num-gpus 8 \
  --decoder-device 0,1,2,3,4,5,6,7 \
  --image-size auto \
  --diff-infer-steps 50 \
  --bot-task vanilla \
  --use-system-prompt en_vanilla \
  --n-samples-per-prompt 4
```

### Output structure

The pipeline first exports latent representations, then decodes them into images:

```text
output_dir/
|-- latents/
|   |-- results.csv
|   `-- *.pt
`-- images/
    `-- *.png
```

## 🧩 Reconstruction

`UniCom_Decoder` also supports reconstruction directly from input images.

### Reconstruction demo

```bash
bash UniCom_Decoder/scripts/run.sh \
  --config-file UniCom_Decoder/configs/reconstruction_demo.yaml
```

The demo images are stored in `UniCom_Decoder/assets/demo_recon_imgs/`.

Each saved output is a side-by-side comparison:

- left: input image
- right: reconstructed image

### Recommended reconstruction settings

The default demo config already uses the recommended settings:

- `mode: eval_gt`
- `aba_mode: compression_64_siglip`
- `condition_mode: siglip2`
- `cfg_scale: 1.0`
- `infer_steps: 50`
- `flow_shift: 3.0`
- `siglip2_max_num_patches: 1024`

## 🙏 Acknowledgement

This project builds upon several excellent open-source projects and research efforts.

- [Tencent HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)
- [Transformers](https://github.com/huggingface/transformers)
- [Diffusers](https://github.com/huggingface/diffusers)
- [FLUX](https://github.com/black-forest-labs/flux)
- [Hugging Face](https://huggingface.co/)


## 📖 Citation

If you find UniCom useful for your research, please cite:

```bibtex
@misc{zhao2026unicomunifiedmultimodalmodeling,
  title={UniCom: Unified Multimodal Modeling via Compressed Continuous Semantic Representations},
  author={Yaqi Zhao and Wang Lin and Zijian Zhang and Miles Yang and Jingyuan Chen and Wentao Zhang and Zhao Zhong and Liefeng Bo},
  year={2026},
  eprint={2603.10702},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.10702},
}
```
