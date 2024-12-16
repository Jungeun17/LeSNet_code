## Beyond Surface Text : Lexical -> Semantic Network for Causal Video Question Answering

**Abstract**

Current models often struggle with choosing distractors based on lexical similarity rather than semantic accuracy, indicating an over-reliance on superficial linguistic features. To address this issue, we propose the **Lexical->Semantic Network (LeSNet)**, a novel contrastive learning approach designed to reduce dependency on lexical cues and enhance detection of subtle details crucial for identifying the correct answer. LeSNet strategically selects hard negative samples using a scoring function that evaluates superficial textual similarities, such as the BLEU score. It also incorporates a sophisticated cross-attention mechanism that dynamically integrates features from the question, each answer candidate, and corresponding video content, utilizing cues from both questions and answers.

![스크린샷 2024-12-16 오후 1 09 56](https://github.com/user-attachments/assets/db0cac8c-bd9d-4e7e-b033-c3daa6292b7c)   


*Note : our repository is mainly based on [<u>CoVGT</u>](https://github.com/doc-doc/CoVGT)   


## Preparation
Download dataset in [https://github.com/bcmi/Causal-VidQA
](https://github.com/bcmi/Causal-VidQA)

## Training 
#### 1. Training the model

```
./shell/cvid_train.sh 0
```

#### 2. Evaluating the model

```
python eval_next.py --folder save_models --mode test
```

 
