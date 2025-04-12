# Mitigating the Bias of Large Language Model Evaluation

This is the official repository for paper [Mitigating the Bias of Large Language Model Evaluation](https://arxiv.org/abs/2409.16788).

In this paper, we propose systematic research about the bias of LLM-as-a-Judge. Specifically, for closed-source judge models, we apply calibration to mitigate the significance of superficial quality, both on probability level and prompt level. For open-source judge models, we propose to mitigate the bias by contrastive training, with curated negative samples that deviate from instruction but present better superficial quality.

## Online Mitigation by Calibration

![](assets\superficial-quality.png)

## Offline Mitigation by Contrastive Training

![](assets\train.png)

## Citation

```bibtex
@inproceedings{zhou2024mitigating,
    title={Mitigating the Bias of Large Language Model Evaluation},
    author={Zhou, Hongli and Huang, Hui and Long, Yunfei and Xu, Bing and Zhu, Conghui and Cao, Hailong and Yang, Muyun and Zhao, Tiejun},
    booktitle={The 23rd China National Conference on Computational Linguistics},
    year={2024}
}
```

## Acknowledgement

This repo benefits from [JudgeLM](https://github.com/baaivision/JudgeLM) and [LLMBar](https://github.com/princeton-nlp/LLMBar). Thanks for their wonderful works.
