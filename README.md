# DeepDiff-SHAP
Interpretable Deep Learning for Subgroup-Specific Causal Inference Using Conditional SHAP-based estimation

Precision medicine aims to tailor healthcare strategies to individual differences in genetic, clinical, and environmental factors. However, identifying subgroup-specific causal relationships in complex biomedical data remains a major challenge, especially when standard causal inference methods average over population heterogeneity. We introduce DeepDiff-SHAP, a novel framework that combines regression-based and deep learning-based differential causal inference to detect changes in causal relationships across patient subgroups. DeepDiff-SHAP integrates SHapley Additive exPlanations (SHAP) to estimate conditional dependencies and quantify differential effects in a principled, interpretable manner. Through a three-step process—detecting structural shifts, testing invariance, and evaluating directional influence via residual variance—we capture nonlinear and high-order causal differences between biological states. Applying DeepDiff-SHAP to two population-scale datasets, the CDC Diabetes Health Indicators Dataset and a UK Biobank sepsis cohort stratified by hypertension, we identified clinically meaningful and subgroup-specific causal changes in relationships between features such as age, general health, alkaline phosphatase, and cholesterol. Our results demonstrate that deep learning enhances sensitivity to complex interaction patterns overlooked by linear models, providing new biological insights into disease progression and comorbidity-specific risk mechanisms. DeepDiff-SHAP offers a scalable and interpretable solution to uncover individualized causal pathways, advancing the goal of truly personalized medicine.


## Installation
The .ipynb has been tested with Python 3.11.5. 

| Parameter            | Info                                                    |
|----------------|----------------------------------------------------------|
| Python version | 3.11.5                                                    |
| Platform       | Linux 3.10.0-1160.71.1.el7.x86_64 (x86_64)                 |
| Hostname       | htc-n68.crc.pitt.edu                                      |

## Install base dependencies
```
git clone https://github.com/ads303/DeepDiff-SHAP.git && cd DeepDiff-SHAP
pip3 install -r requirements.txt
```

## Setup
The example in the .ipynb file uses the UCI Diabetes Health Indicators Dataset. There is a commented block that uses example data, which should guide how you format the dataset of choice that you end up using. You can use the UCI Diabetes data, the example data, or your own data to test DeepDiff-SHAP.


## Disclaimer
I have no external affiliations. 

