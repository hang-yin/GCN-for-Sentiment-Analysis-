# CS397 Spring 2022
This final project explores combining dependency parsing with pre-trained language embedding models using edge-conditioned graph convolution networks.

By: Joey Yang, Hang Ying, Renzhi Hao, and Junhao Xu

## Abstract:

### Purpose:
SOTA sentiment classification approaches rely on very large statistical language models that perform well but difficult to interpret. By training and representing language models as graphs with separable semantic and syntactical features, we can better understand how language models make decisions and build models that reflect human understanding of language. 

### Methods:
The theory of dependency grammar represents the syntactical structure of sentences as a directed graph where the nodes as words and the edges are relations between the words. Each edge describes how the target word, named the dependent, modifies the source word, named the head. By combining dependency trees and pre-trained language models, we can represent sentences as trees where the nodes are n-dimensional vector representations of words, and the edges are the dependency relations. By using Edge-Conditioned Graph Convolution Networks, we aim to combine the structural, syntactic, and semantic information from a sentence for the task of sentiment classification. 

### Results:
Edge-Conditioned GCN (87.36) outperforms baselines. However, one-hot encoding of edge types is insufficient for learning or utilizing different types of depenendcy relations.


### Conclusion:
We demonstrated that it is possible to perform sentiment classification while integrating syntactical and semantic information in a graphical way. However, the task of learning and integrating the representations of dependency relation types is still unsolved.

