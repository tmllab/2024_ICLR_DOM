<div align="center">

# On the Over-Memorization During Natural, Robust and Catastrophic Overfitting
[![Paper](https://img.shields.io/badge/paper-ICLR-green)]()

</div>

Official implementation of [On the Over-Memorization During Natural, Robust and Catastrophic Overfitting]() (ICLR 2024).

## Abstract
Overfitting negatively impacts the generalization ability of deep neural networks (DNNs) in both natural and adversarial training. Existing methods struggle to consistently address different types of overfitting, typically designing strategies that focus separately on either natural or adversarial patterns. In this work, we adopt a unified perspective by solely focusing on natural patterns to explore different types of overfitting. Specifically, we examine the memorization effect in DNNs and reveal a shared behaviour termed over-memorization, which impairs their generalization capacity. This behaviour manifests as DNNs suddenly becoming high-confidence in predicting certain training patterns and retaining a persistent memory for them. Furthermore, when DNNs over-memorize an adversarial pattern, they tend to simultaneously exhibit high-confidence prediction for the corresponding natural pattern. These findings motivate us to holistically mitigate different types of overfitting by hindering the DNNs from over-memorization training patterns. To this end, we propose a general framework, $\textit{Distraction Over-Memorization} (DOM)$, which explicitly prevents over-memorization by either removing or augmenting the high-confidence natural patterns. Extensive experiments demonstrate the effectiveness of our proposed method in mitigating overfitting across various training paradigms.

## Requirements
- This codebase is written for `python3` and 'pytorch'.
- To install necessary python packages, run `pip install -r requirements.txt`.


## Experiments
### Data
- Please download and place all datasets into the data directory.


### Training

Standard Training
```
python3 DOM.py --lr-max 0.1 --epochs 300 --attack none --operate RE --clamp 0.2 --max-arg-iteration 0 --arg-strength 0.0 --fname Natural_DOM_RE
python3 DOM.py --lr-max 0.1 --epochs 300 --attack none --operate DA_AUG --clamp 0.2 --max-arg-iteration 3 --arg-strength 0.5 --fname Natural_DOM_A
python3 DOM.py --lr-max 0.1 --epochs 300 --attack none --operate DA_Rand --clamp 0.2 --max-arg-iteration 3 --arg-strength 0.1 --fname Natural_DOM_R
```

Multi-step Adversarial Training
```
python3 DOM.py --lr-max 0.1 --epochs 200 --attack pgd --operate RE --clamp 1.5 --max-arg-iteration 0 --arg-strength 0.0 --fname PGD_DOM_RE
python3 DOM.py --lr-max 0.1 --epochs 200 --attack pgd --operate DA_AUG --clamp 1.5 --max-arg-iteration 2 --arg-strength 0.5 --fname PGD_DOM_A
python3 DOM.py --lr-max 0.1 --epochs 200 --attack pgd --operate DA_Rand --clamp 1.5 --max-arg-iteration 2 --arg-strength 0.0 --fname PGD_DOM_R
```

Single-step Adversarial Training
```
python3 DOM.py --lr-max 0.2 --epochs 100 --attack fgsm --operate RE --clamp 2.0 --max-arg-iteration 0 --arg-strength 0.00 --fname FGSM_DOM_RE
python3 DOM.py --lr-max 0.2 --epochs 100 --attack fgsm --operate DA_AUG --clamp 2.0 --max-arg-iteration 5 --arg-strength 0.5 --fname FGSM_DOM_A
python3 DOM.py --lr-max 0.2 --epochs 100 --attack fgsm --operate DA_Rand --clamp 2.0 --max-arg-iteration 3 --arg-strength 0.1 --fname FGSM_DOM_R
```


## License and Contributing
- This README is formatted based on [paperswithcode](https://github.com/paperswithcode/releasing-research-code).
- Feel free to post issues via Github.

## Reference
If you find the code useful in your research, please consider citing our paper:

<pre>
@inproceedings{lin2023over,
  title={On the Over-Memorization During Natural, Robust and Catastrophic Overfitting},
  author={Lin, Runqi and Yu, Chaojian and Han, Bo and Liu, Tongliang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
</pre>
