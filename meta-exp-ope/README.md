# Counterfactual Model Selection in Contextual Bandits

This repository contains the code for the paper "Counterfactual Model Selection in Contextual Bandits," accepted to short paper track of SIGIR 2025.


## Abstract

Contextual bandit algorithms are crucial in various decision-making applications, such as personalized content recommendation, online advertising, and e-commerce banner placement. Despite their successful applications in various domains, contextual bandit algorithms still face significant challenges with exploration efficiency compared to non-contextual bandit algorithms due to exploration in feature spaces. To overcome this issue, model selection policies such as MetaEXP and MetaCORRAL have been proposed to interactively explore base policies. In this paper, we introduce a novel counterfactual approach to address the model selection problem in contextual bandits. Unlike previous methods, our approach leverages unbiased Off-Policy Evaluation (OPE) to dynamically select base policies, making it more robust to model misspecification. We present two new algorithms, MetaEXP-OPE and MetaGreedy-OPE, which utilize OPE for model selection policy. We also provide theoretical analysis on regret bounds and evaluate the impact of different OPE estimators. We evaluated our model on synthetic data and a semi-synthetic simulator using a real-world dataset, and the results show that MetaEXP-OPE and MetaGreedy-OPE significantly outperform existing policies, including MetaEXP and MetaCORRAL.

## Installation
```
git clone [repository URL]
cd meta-exp-ope
```

Plese clone [zr_obp](https://github.com/st-tech/zr-obp).
and put `model_selection.py` on `obp/policy` dir.

