## Model Download

### Huggingface

| Model                 | Introduction | Download                                                                    |
|-----------------------|-----------------|-----------------------------------------------------------------------------|
| Janus-Pro-CXR-Zero(1B) | Finetuned on MIMIC-CXR dataset with Janus-Pro-1B    | [🤗 Hugging Face](https://huggingface.co/ZrH42/Janus-Pro-CXR-Zero) |
| Janus-Pro-CXR-Final(1B) | Finetuned on CXR-27 dataset with Janus-Pro-CXR-Zero | [🤗 Hugging Face](https://huggingface.co/ZrH42/Janus-Pro-CXR-Final) |
| Janus-Pro-1B | Official model from deepseek | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/Janus-Pro-1B) |
| Janus-Pro-7B | Official model from deepseek | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/Janus-Pro-7B) |

## Quick Start

### Installation

On the basis of `Python >= 3.8` environment, install the necessary dependencies by running the following command:

```shell
conda create -n Janus-Pro-CXR python=3.10
conda activate Janus-Pro-CXR
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


### Inference Example

```shell
python inference.py ./Janus-Pro-CXR-Final ./retrospective_data/2.png
python inference.py /path/to/your/model /path/to/your/image
```

You can try feeding the results of a expert diagnostic model into the prompt.

```shell
python inference.py ./Janus-Pro-CXR-Final ./retrospective_data/2.png --prompt "Expert analysis: Lung Opacity, Pleural Effusion, Pleural Other, Pneumothorax, Support Devices present.Please examine this X-ray and provide a radiology report including FINDINGS and IMPRESSION."
```

For Sample 1, the comparative results across three paradigms are presented below.

|                |                        gold standard                         |                         unify model                          |                  unify model + expert model                  |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     report     | FINDINGS: postoperative changes in the right lung with visualization of the drainage tube. Minimal pneumothorax is noted in the right pleural cavity. Patchy opacities are scattered in the right lung field and the lower left lung field. The trachea and mediastinum are midline. The cardiac silhouette shows no significant abnormalities in size or shape. The costophrenic angles are slightly obscured bilaterally. Subcutaneous emphysema is present on the right side. IMPRESSION: 1. Postoperative status of the right lung with pneumothorax on the right side; 2. Patchy opacities scattered in the right lung field and lower left lung field, CT examination recommended if necessary; 3. Minimal bilateral pleural effusion. | FINDINGS: postoperative changes in the right lung with a drainage tube visible in the right thoracic cavity. A few linear and patchy opacities are noted in the lower lung fields bilaterally. The trachea and mediastinum are midline. The cardiac silhouette appears normal in size and configuration. Both diaphragmatic surfaces are slightly obscured, with blunting of the left costophrenic angle. IMPRESSION: 1. Postoperative status of the right lung; clinical correlation and follow-up are recommended. 2. A few linear and patchy opacities in the lower lung fields bilaterally, possibly representing atelectasis; follow-up is advised. 3. Small bilateral pleural effusions. | FINDINGS: postoperative changes in the right lung with a small amount of pneumothorax on the right side, showing minimal subcutaneous emphysema in the right chest wall; a drainage tube is noted in the right pleural cavity. A few linear and patchy opacities are observed in both lung fields. The trachea and mediastinum are midline, and the cardiac silhouette is not enlarged. The costophrenic angles are blunted bilaterally. IMPRESSION: 1. Postoperative status of the right lung; minimal subcutaneous emphysema in the right chest wall. 2. A few linear and patchy opacities in both lung fields, possibly due to inflammation/segmental atelectasis. 3. Bilateral small pleural effusions/pleural adhesions. 4. Minimal pneumothorax on the right side. |
| deepseek label | Lung Opacity，Pneumothorax，Pleural Effusion，Support Devices | Lung Opacity，Atelectasis，Pleural Effusion，Support Devices | Lung Opacity，Atelectasis，Pneumothorax，Pleural Effusion，Pleural Other，Support Devices |

#### DeepSeek labeling tool

First, obtain and update the DeepSeek API key in the code. Then, run the provided code.

```shell
python label_with_deepseek.py --prompt "FINDINGS: postoperative changes in the right lung with visualization of the drainage tube. Minimal pneumothorax is noted in the right pleural cavity. Patchy opacities are scattered in the right lung field and the lower left lung field. The trachea and mediastinum are midline. The cardiac silhouette shows no significant abnormalities in size or shape. The costophrenic angles are slightly obscured bilaterally. Subcutaneous emphysema is present on the right side. IMPRESSION: 1. Postoperative status of the right lung with pneumothorax on the right side; 2. Patchy opacities scattered in the right lung field and lower left lung field, CT examination recommended if necessary; 3. Minimal bilateral pleural effusion."
```

The expected output is as follows:

```json
{
  "Enlarged Cardiomediastinum": 0,
  "Cardiomegaly": 0,
  "Lung Opacity": 1,
  "Lung Lesion": 0,
  "Edema": 0,
  "Consolidation": 0,
  "Pneumonia": 0,
  "Atelectasis": 0,
  "Pneumothorax": 1,
  "Pleural Effusion": 1,
  "Pleural Other": 0,
  "Fracture": 0,
  "Support Devices": 1,
  "No Finding": 0
}
```

## Citation

```bibtex

```
