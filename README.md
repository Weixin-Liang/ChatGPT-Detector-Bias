# GPT Detectors Are Biased Against Non-Native English Writers

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

This repository contains the data and supplementary materials for our paper:

**GPT Detectors Are Biased Against Non-Native English Writers**\
Weixin Liang*, Mert Yuksekgonul*, Yining Mao*, Eric Wu*, James Zou\
arXiv: [2304.02819](https://arxiv.org/abs/2304.02819)

```bibtex
@article{liang2023gpt,
    title={GPT detectors are biased against non-native English writers}, 
    author={Weixin Liang and Mert Yuksekgonul and Yining Mao and Eric Wu and James Zou},
    year={2023},
    eprint={2304.02819},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Abstract
*The rapid adoption of generative language models has brought about substantial advancements in digital communication, while simultaneously raising concerns regarding the potential misuse of AI-generated content. Although numerous detection methods have been proposed to differentiate between AI and human-generated content, the fairness and robustness of these detectors remain underexplored. In this study, we evaluate the performance of several widely-used GPT detectors using writing samples from native and non-native English writers. Our findings reveal that these detectors consistently misclassify non-native English writing samples as AI-generated, whereas native writing samples are accurately identified. Furthermore, we demonstrate that simple prompting strategies can not only mitigate this bias but also effectively bypass GPT detectors, suggesting that GPT detectors may unintentionally penalize writers with constrained linguistic expressions. Our results call for a broader conversation about the ethical implications of deploying ChatGPT content detectors and caution against their use in evaluative or educational settings, particularly when they may inadvertently penalize or exclude non-native English speakers from the global discourse.*


<p align='center'>
  <img width="636" src="https://user-images.githubusercontent.com/32794044/230640445-8d1221d4-8651-4cf4-b6d7-b6d440d6e0f5.png">
  <br>
  <b>Figure 1: Bias in GPT detectors against non-native English writing samples.</b>
</p>
(a) Performance comparison of seven widely-used GPT detectors. More than half of the non-native-authored TOEFL (Test of English as a Foreign Language) essays are incorrectly classified as "AI-generated," while detectors exhibit near-perfect accuracy for college essays.
Using ChatGPT-4 to improve the word choices in TOEFL essays (Prompt: "Enhance the word choices to sound more like that of a native speaker.") significantly reduces misclassification as AI-generated text.
(b) TOEFL essays unanimously misclassified as AI-generated show significantly lower perplexity compared to others, suggesting that GPT detectors might penalize authors with limited linguistic expressions.




<p align='center'>
  <img width="100%" src="https://user-images.githubusercontent.com/32794044/230640270-e6c3d0ca-aabd-4d13-8527-15fed1491050.png">
  <br>
  <b>Figure 2: Simple prompts effectively bypass GPT detectors.</b>
</p>
(a) For ChatGPT-3.5 generated college admission essays, the performance of seven widely-used GPT detectors declines markedly when a second-round self-edit prompt ("Elevate the provided text by employing literary language") is applied, with detection rates dropping from up to 100% to up to 13%.
(b) ChatGPT-3.5 generated essays initially exhibit notably low perplexity; however, applying the self-edit prompt leads to a significant increase in perplexity.
(c) Similarly, in detecting ChatGPT-3.5 generated scientific abstracts, a second-round self-edit prompt ("Elevate the provided text by employing advanced technical language") leads to a reduction in detection rates from up to 68% to up to 28%.
(d) ChatGPT-3.5 generated abstracts have slightly higher perplexity than the generated essays but remain low. Again, the self-edit prompt significantly increases the perplexity.



## Repo Structure Overview


```
.
├── README.md
├── data/
    ├── human_data/ 
        ├── TOEFL_real_91/
            ├── name.json
            ├── data.json
        ├── TOEFL_gpt4polished_91/
            ├── ...
        ├── CollegeEssay_real_70/
        ├── CS224N_real_145/
    ├── gpt_data/ 
        ├── CollegeEssay_gpt3_31/
        ├── CollegeEssay_gpt3PromptEng_31/
        ├── CS224N_gpt3_145/
        ├── CS224N_gpt3PromptEng_145/
```


The `data` folder contains the human-written and AI-generated datasets used in our study. Each subfolder contains a `name.json` file, which provides the metadata, and a `data.json` file, which contains the text samples.



## Reference
```bibtex
@article{liang2023gpt,
    title={GPT detectors are biased against non-native English writers}, 
    author={Weixin Liang and Mert Yuksekgonul and Yining Mao and Eric Wu and James Zou},
    year={2023},
    eprint={2304.02819},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```