# Edge-Conditioned Graph Convolution Networks for Sentiment Analysis
This project explores combining dependency parsing with pre-trained language embedding models using edge-conditioned graph convolution networks.

## Purpose
SOTA sentiment classification approaches rely on very large statistical language models that perform well but difficult to interpret. By training and representing language models as graphs with separable semantic and syntactical features, we can better understand how language models make decisions and build models that reflect human understanding of language. 

## Methods
The theory of dependency grammar represents the syntactical structure of sentences as a directed graph where the nodes as words and the edges are relations between the words. See below for an example of a dependency graph: 

![dependency-parsing](https://user-images.githubusercontent.com/60046203/230534488-8bdb0cc0-79fc-446e-8c5e-541a082018f2.png)

Each edge describes how the target word, named the dependent, modifies the source word, named the head. By combining dependency trees and pre-trained language models, we can represent sentences as trees where the nodes are n-dimensional vector representations of words, and the edges are the dependency relations. By using Edge-Conditioned Graph Convolution Networks, we aim to combine the structural, syntactic, and semantic information from a sentence for the task of sentiment classification. See below for our model architecture: 

<img width="722" alt="nlp_pipeline" src="https://user-images.githubusercontent.com/60046203/230534505-c37973af-7578-44ad-b2ea-bb21e5d4ace7.png">

## Results
Edge-Conditioned GCN (87.36%) outperforms baselines on the IMDB binary classification task with significantly less trainable as well as fixed parameters. However, one-hot encoding of edge types is insufficient for learning or utilizing different types of depenendcy relations.

| Model                                        | Trainable parameters | Total parameters | Accuracy |
|----------------------------------------------|----------------------|------------------|----------|
| Feed Forward                                 | 8k                   | 47M              | 76.30%   |
| BiLSTM                                       | 300k                 | 47M              | 84.73%   |
| CNN                                          | 7.9M                 | 55M              | 84.50%   |
| GCN sentence-level                           | 33k                  | 47M              | 65.68%   |
| GCN with dependency parsing                  | 561                  | 500k             | 85.32%   |
| Edge-conditioned GCN with dependency parsing | 20k                  | 480k             | 87.36%   |

## Conclusion
We demonstrated that it is possible to perform sentiment classification while integrating syntactical and semantic information in a graphical way. We also
explored both a standard graph convolution layer and an edge-conditioned convolution layer that incorporates dependency relations as edge labels.

## Link to paper
Check out our paper [here](https://raw.githubusercontent.com/hang-yin/portfolio/gh-pages/assets/nlp-paper.pdf).

## Group members
Hang Yin, Joey Yang, Junhao Xu, Renzhi Hao

