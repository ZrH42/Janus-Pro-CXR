## Model Download

### Huggingface

| Model                 | Introduction | Download                                                                    |
|-----------------------|-----------------|-----------------------------------------------------------------------------|
| Janus-Pro-CXR-Zero(1B) | Finetuned on MIMIC-CXR dataset with Janus-Pro-1B    | [🤗 Hugging Face](https://huggingface.co/ZrH42/Janus-Pro-CXR-Zero) |
| Janus-Pro-CXR-Final(1B) | Finetuned on CXR-27 dataset with Janus-Pro-CXR-Zero | [🤗 Hugging Face](https://huggingface.co/ZrH42/Janus-Pro-CXR-Final) |
| Janus-Pro-1B | Naive model from deepseek | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/Janus-Pro-1B) |
| Janus-Pro-7B | Naive model from deepseek | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/Janus-Pro-7B) |

## Quick Start

### Installation

On the basis of `Python >= 3.8` environment, install the necessary dependencies by running the following command:

```shell
conda create -n Janus-Pro-CXR python=3.10
conda activate Janus-Pro-CXR
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements
```


### Inference Example

```shell
python inference.py ./Janus-Pro-CXR-Final ./retrospective_data/1.png
python inference.py /path/to/your/model /path/to/your/image
```

## Citation

```bibtex

```
