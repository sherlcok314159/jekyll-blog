---
title: Retrieval-Enhanced Transformer
date: 2022-06-19 21:06:00 +0800
categories: [PLMs]
tags: [retrieval, transformer]
math: true
mermaid: false
image:
  path: /2022/06/19/Xvif95kmEYgHupM.jpg
  width: 600
  height: 100
---

## Problems To Solve
1. To <b>Scale Down</b> the model size while maintaining the performances.
2. To incorporate `External Memory Retrieval` in the Large Language Model Modelling.

![](2022/06/19/ecSmGwTuBbzYnDX.png)

## How?

### Data Construction
1. Training & Evaluation set:
	1. $\text{MassiveText}$ for both training & retrieval data (contains 5 trillion tokens)
	![](2022/06/19/JUpDF8y9LCqnRNW.png)
	2. `SentencePiece` with a vocabulary of $128K$ tokens
	3. During training, we retrieving $600B$ tokens from the training 
	4. The evaluation contains $1.75T$ tokens
2. `Test set` leakage:
	Due to the huge retrieving database, the test set may have appeared in the training set. Thus, the authors apply `13-gram Jaccard Similarity` between the training and test documents to *filter* those training documents similar to the test documents (i.e., the similarity is $\geq \textbf{0.80}$)

### Retrieval Modelling
1. `Key-Value` Format of the Database: 
	1. $\text{Key} \Rightarrow$ `frozen` BERT Embedding
	2. $\text{Value} \Rightarrow$ raw `chunks` of the tokens
2. using the `SCaNN` library 
3. the similarity depends on the $\text{L2 Distance}$:

	$$
	||x-y||_2  = \sqrt{\sum_i (x_i - y_i)^2}
	$$
4. **pre-compute** the `frozen` BERT Embedding to save the computation and the Embedding is **averaged with time.**
5. retrieving targets are the corresponding chunks and their `continuation` in the orig document

### The whole architecture

#### The pipeline

![](2022/06/19/SMKJbATvzyqRE3c.png){: w="500" h="800" }

1. Assume the input sequence $\text{X}$ contains $9$ tokens, it can be `split` into $3$ chunks (i.e., $C_1, C_2, C_3$) whose sizes are $3$ respectively.
2. Then the chunks are embedded through the frozen BERT embedding. We can retrieve neighbours of those input chunks.
3. We also embed the input sequence and then apply `self-attention mechanism` on them to get the hidden states $H(X)$
4. Furthermore, we need to encode the neighbours. Here, the transformer encoder is `bi-directional`. And it outputs the representations of the neighbours by `conditioning` on the hidden states of the input chunks.
5. After we get the representations of the neighbours, we let them `attend` the input chunks as the $\text{K and V}$ while the input chunk is $\text{Q}$. The attending network is called CCA($\textbf{C}$hunked $\textbf{C}$ross $\textbf{A}$ttention). I introduce it in the following part.
6. When the neighbours finish attending the input chunks, the input chunks can be `represented` by the retrieved neighbours. The representations are going through the FFW($\textbf{F}$eed $\textbf{F}$or$\textbf{W}$ard). Thus, a `Retro-Block` contains self-attention mechanism, CCA and FFW.

#### Chunked Cross Attention

![](2022/06/19/EJ4GsShCHox2dmn.png){: w="500" h="800" }

1. Take the green chunk as the example, we retrieve its neighbours from the database and we let them attend with the `concatenation` between the green chunk and its next chunk. To put it more precisely, assume we retrieve the neighbours $E(m_i)$ for the chunk $m_i$ which contains $n$ tokens: ${m_{i1}, m_{i2}, \dots, m_{in}}$, we concatenate the `last` token of $m_i$ with the `next` chunk $m_j$ `except` the last token $\Rightarrow \text{Concatenate}(m_{in}, m_{j1, \dots, jn-1})$. 
2. After the concatenation, we apply `CA`($\textbf{C}$ross $\textbf{A}$ttention). CA is the common attention mechanism.
3. Finally, we `concatenate` the outputs and `pad` them.

> Note, the relative positional encoding is applied.
{: .prompt-tip }

## Experiment

### Scaling the Retro
![](2022/06/19/2Pl7NrxWuyzkEwO.png)

![](2022/06/19/DzKGVWuH2r4Tvdj.png)
1. The `scale` of the Retro and the retrieved `tokens` are `proportional` to the performance.
2. The number of neighbours has an `upped bound`: somewhere near $40$. Maybe too many neighbours `reduce` the retrieval quality.

### Improvement Comparison
![](2022/06/19/yeOLafErnChsq3K.png)
1. Among some tasks, Retro can `outperform` the models whose parameters are much more than the Retro's. 

### Perplexity on Wikitext103
![](2022/06/19/Fueq18xZWtO5CwV.png)
1. Retro's perplexity can be `SOTA` on the Wikitext103
2. Interestingly, the external memory can also have the phenomenon of the `underfitting`. When using `MassiveText(1%)`, it can underfit the training set. And its performance is worse than the `kNN-LM`.

### Retro Finetuning
![](2022/06/19/oBCmrJ5XPqZuezL.png)
1. `Training from scratch` is the most powerful way.

### Question Answering Results
![](2022/06/19/9NX3QnemqWwY7oC.png){: w="500" h="400" }
1. `FID + Distill` is the `SOTA` in the Open-Domain Question Answering when the retrieval involves in the training.

### Ablation Studies
![](2022/06/19/WaxCPTmzJSw8dDi.png){: w="500" h="800" }
1. The `continuation` of the retrieved chunks do `help`.
2. CA positions are `every 3 from 1 or mid layer`.

## Why work?
1. To summarize,  the Retro incorporates the `external` neighbours of the input sequence into the Large Language Modelling to `scale down` the model size while `maintaining` the performance.

## Lessons & Imaginations
1. Performance can get improved either by improving the `model size` or training `more data`.
2. Huge amount of data `don't` need too `big` model to fit in.
3. We can scale down the PLM by attending the `external information`.
4. CCA is applied because the external knowledge need to be merged. When applying in `MRC`, the `external` information can be:
	1. the chunked passages
	2. the broken passages
	3. the past similar to question-passage pairs
	4. the knowledge among the input
	5. the evidence
5. The `BM25, Edit Distance and LDA` can also perform not bad in the retieval.
