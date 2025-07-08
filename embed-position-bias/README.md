# "Position Bias Estimation with Item Embedding for Sparse Dataset" 
## Introduction
This repository contains the implementation of the methods described in the paper "Improving Position Bias Estimation Against Sparse and Skewed Dataset with Item Embedding". The paper addresses the challenge of estimating position bias in Learning to Rank (L2R) systems, particularly in e-commerce applications.

## Paper Abstract
Estimating position bias is a well-known challenge in Learning to rank (L2R).  Click data in e-commerce applications, such as advertisement targeting and search engines, provides implicit but abundant feedback to improve personalized rankings. However, click data inherently include various biases like position bias. Click modelling is aimed at denoising biases in click data and extracting reliable signals. Result Randomization and Regression Expectation-maximization algorithm (REM) have been proposed to estimate position bias. Both methods require various paired observations, consisting of (item, position). However, in real cases of advertising, marketers frequently display advertisements in a fixed pre-determined order, and estimation suffers from it. We propose this sparsity of (item, position) in position bias estimation as a novel problem, and we propose a variant of the REM which utilizes item embeddings to alleviate this issue of sparsity. With a public dataset and internal real traffic of carousel advertisements, we empirically show that item embedding with Latent Semantic Indexing (LSI) and Variational auto-encoder (VAE) improves the accuracy of position bias estimation. In addition, our results show the estimated position bias improves the performance of learning to rank. We also show that LSI is more effective as the item embedding for position bias estimation.

## Repository Structure
```
src/: Source code for the implementation of the proposed method.
data/: Sample datasets used for evaluation.
notebooks/: Jupyter notebooks demonstrating the usage and results.
```

## Setup and Installation
Instructions for setting up the environment, including installing required libraries and dependencies.
```
git clone git@ghe.rakuten-it.com:aippm-customer/embed_position_bias.git
cd embed_position_bias
```

Please install Open Bandit Dataset and store in `data` dir.
- https://github.com/st-tech/zr-obp


## Citation
If you use this implementation in your research, please cite the following paper:

```
@misc{ishikawa2024position,
      title={Position Bias Estimation with Item Embedding for Sparse Dataset}, 
      author={Shion Ishikawa and Yun Ching Liu and Young-Joo Chung and Yu Hirate},
      year={2024},
      eprint={2305.13931},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```


## Contact
Details for contacting the authors or maintainers of the repository for any queries or contributions.
- shion.ishikawa@rakuten.com
