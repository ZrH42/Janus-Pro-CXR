import os
import json
import re
from openai import OpenAI

client = OpenAI(
    api_key="sk-***********************", # Your api key
    base_url="https://api.deepseek.com"
)

findings = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]

def analyze_findings(report: str) -> dict:
    prompt = f"""
    Based on the following chest X-ray report, determine the presence of the 14 listed findings. Return 1 if present, 0 if absent.
    Reply only with a JSON object containing each finding and its corresponding 0 or 1.

    Report:
    {report}

    Findings:
    Enlarged Cardiomediastinum: abnormal enlargement of cardiac and mediastinal contours
    Cardiomegaly: enlarged cardiac silhouette, increased cardiothoracic ratio
    Lung Opacity: increased density areas within lung fields
    Lung Lesion: pulmonary nodules, masses, or other focal lesions
    Edema: pulmonary edema, typically presenting as interstitial or alveolar infiltrates
    Consolidation: pulmonary consolidation, uniformly increased density
    Pneumonia: findings suggestive of infectious pneumonia such as bronchitis or inflammation
    Atelectasis: collapsed lung tissue with reduced lung volume
    Pneumothorax: air in the pleural cavity
    Pleural Effusion: fluid accumulation in the pleural cavity
    Pleural Other: other pleural abnormalities
    Fracture: fractures of thoracic bones such as ribs
    Support Devices: medical devices like endotracheal tubes, pacemakers, central venous catheters
    No Finding: none of the above abnormalities present, normal or no described abnormalities
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a professional radiologist specialized in chest X-ray diagnosis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    text = response.choices[0].message.content
    match = re.search(r'\{.*\}', text.replace('\n', ''), re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            for f in findings:
                result.setdefault(f, 0)
            return result
        except json.JSONDecodeError:
            pass
    return {f: 0 for f in findings}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze chest X-ray report")
    parser.add_argument("--prompt", type=str, required=True, help="Input medical report text")
    args = parser.parse_args()
    print(json.dumps(analyze_findings(args.prompt), ensure_ascii=False, indent=2))