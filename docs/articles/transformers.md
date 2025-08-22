# **Transformer**

A **Transformer** is a class of neural networks defined by a common architectural design principle.  
Its defining component is the **self-attention** mechanism—typically implemented as **multi-head attention**.  
Since the seminal paper *“Attention Is All You Need”* (Vaswani et al., 2017), this mechanism has proven highly effective for modeling token-to-token dependencies under context and has become a viable alternative to recurrent architectures (RNNs).

Today, the Transformer architecture underpins most state-of-the-art models for text, images, audio, and multimodal data.  
Models such as **GPT**, **BERT**, **LLaMA**, and the **Vision Transformer (ViT)** are all based on this architecture.  
Consequently, **transformer-based models**—including toy implementations—are the canonical objects of study in mechanistic interpretability.

## **Dynamic Approach to Embedding Formation** 

In classical text processing models, a **word embedding** is static: each word is always mapped to the same vector, regardless of its context.  
While effective in statistical models, this approach has a fundamental limitation: the same word can take on different meanings depending on context.  

**Example: Homonymy:**

- *“The wooden **lock** was hard to open.”*  
- *“There was no one in the wooden **castle**.”*  

**Example: Ambiguous Pronouns:**
- *“The cat chased the mouse because **it** was hungry.”*  
- *“The cat chased the mouse because **it** was small.”*  

In these examples, the same word or pronoun requires different interpretations depending on the surrounding context.  

To address such cases, the field has moved from **static word embeddings** to **contextual embeddings**,  
since context refines the semantic meaning of words and yields more accurate sentence representations in vector space.  
In Transformers, this shift is realized through the **Attention** mechanism.  

## **Attention: Intuition**

Consider a sentence represented as a sequence of tokens:  

$$[x_1, x_2, \dots, x_{i-1}, x_i, \dots, x_n]$$  

Suppose that in order to form a correct representation of the token $x_i$, we require — and are limited to — the representations of all **preceding tokens**.  


<details>
<summary>**Why can’t we look into the future?**</summary>

If the model were to use future tokens to predict the current one, it would effectively be *peeking at the answer*.  
In the task of **language modeling** (predicting the next word given the preceding words), this would cause information leakage:  
to predict $x_i$, the model must **not** have access to $x_{i+1}$.  
Therefore, attention is constrained to past tokens only.  

</details>


<details>
<summary>**How does this help with pronoun disambiguation?**</summary>

- *“The cat chased the mouse because **it** was hungry.”*  
- *“The cat chased the mouse because **it** was small.”*  

Here, the context determines the meaning of the pronoun **it**.  
By attending **left-to-right**, the model uses the preceding tokens to decide whether **it** refers to the cat or the mouse,  
while the continuation of the sentence refines that interpretation.  
</details>

**Right-to-left Languages** 

There are languages, such as Arabic, where text is written and read from right to left.  
This does not prevent the implementation of attention: the model still processes the sequence in the **logical order** of symbols or sub-tokens as defined for that language.  

## **Attention: Intuition — Query, Key, and Value Roles**  

Let us introduce the idea that during processing, each token can play one of three roles:  

1. **Query ($q$):** the current element compared with all previous ones.  
   *Analogy:* a search query compared to available documents.  

2. **Key ($k$):** a past element compared with the current one.  
   *Analogy:* finding the right key among $x_j, j \leq i$ to fit the current lock $x_i$.  

3. **Value ($v$):** the content of a past element, whose weighted contribution is included in the representation of the current element $x_i$.  
   *Analogy:* the document retrieved for the user once the query matches a key.  

---

To represent these three roles, each token vector $x_i$ is projected into three distinct vectors:  

- $q_i = x_i W^Q$ — the token as a query,  
- $k_i = x_i W^K$ — the token as a key for comparison,  
- $v_i = x_i W^V$ — the token as a value, providing content to the current representation.  

Here, $W^Q, W^K, W^V$ are trainable matrices.  


**Measuring Similarity** 

For each $x_i$, the representation is formed by comparing its query $q_i$ with the keys of previous tokens $k_j, j \leq i$.  
Similarity can be measured by the **dot product**, which has the geometric interpretation of the cosine of the angle between two vectors scaled by their magnitudes:  

$$
\langle a, b \rangle = |a||b|\cos(a, b)
$$  


**Output Representation ** 

Following this logic, the final representation $o_i$ of the token $x_i$ is a weighted combination of value vectors:  

$$
o_i = \alpha_{i1} v_1 + \alpha_{i2} v_2 + \dots + \alpha_{in} v_n
$$  

where the weights $\alpha_{ij}$ depend on the similarity between query $q_i$ and key $k_j$.  


![attn_gig](https://ucarecdn.com/6c902bf8-5761-4e23-a3a9-2e303e17afef/)
Awesome attention gif,  [src](https://medium.com/@incle/attention-mechanism-math-illustration-transformers-series-part-1-37c24ac9d2f2)

## **Attention: Formalization** 

We now formalize the idea from the previous step.  
Initially, each token vector $x_i$ is projected into three components:  

- $q_i = x_i W^Q$, with dimension $[1, d_k]$  
- $k_i = x_i W^K$, with dimension $[1, d_k]$  
- $v_i = x_i W^V$, with dimension $[1, d_v]$  

Note that the dimensions of $q, k, v$ are not equal to the original embedding dimension of the model (denoted $d_{\text{model}}$).  
For example, in the original Transformer paper, the embedding dimension was $d_{\text{model}} = 512$,  
while the dimensions of $q, k, v$ were $d_k = d_v = 64$.  


**Goal:**

For each token $x_i$, we want to form a representation $o_i$ that incorporates the weighted contributions of all previous tokens $x_j, j \leq i$.  
Simplified, each representation $o_i$ should take the form:  

$$
o_i = \sum_{j \leq i} \alpha_{ij} v_j
$$  

where the weights $\alpha_{ij}$ reflect the similarity between the current token $x_i$ and past tokens $x_j$.  

To construct this weighted sum, three steps are necessary:  

1. **Compute similarity scores** for each previous vector.  
2. **Normalize the scores** so that they are positive and comparable  
   (strong positive and negative contributions should not cancel each other out).  
3. **Form the weighted representation** by combining the value vectors according to these normalized weights.  


## **Attention: Formalization 2** 

We now detail the formal computation of attention in several steps.  

**Step 1. Computing Similarity Scores** 

For each previous token $x_j$, define its similarity to the current token $x_i$ as:  

$$
sim_{score}(x_i, x_j) = \frac{\langle q_i, k_j \rangle}{\sqrt{d_k}}
$$  

**Why divide by $\sqrt{d_k}$?**  
Scaling the dot product by $\sqrt{d_k}$ prevents exploding or vanishing gradients.  
Without this normalization, as the embedding dimension $d_k$ grows, dot products become very large (their variance grows with $d_k$).  
In this case, the softmax in the next step produces sharp peaks: almost all weights $\alpha_{ij}$ become close to zero except for one, leading to training instability.  
Scaling keeps the distribution well-behaved and smooths the softmax output.  


![dot_pr](https://ucarecdn.com/36d95c05-213e-4e99-bf38-64841fe5bb49/)
*Dot-product values ​​for ebmeaddings from a standard normal distribution.*

**Step 2. Normalizing Scores** 

All similarity scores are normalized to the interval $[0, 1]$ using the softmax transformation, yielding attention weights:  

$$
\alpha_{ij} = \text{softmax}(sim_{score}(x_i, x_j)), \quad \forall j \leq i
$$  

**Why softmax normalization?**  
Softmax maps all values into $[0, 1]$ and ensures they sum to 1.  
This allows the weights to be interpreted directly as attention coefficients.  
It also removes issues from raw dot products where negative and positive values could cancel each other out.  

![softmax](https://ucarecdn.com/cb688160-7377-41ab-8f9d-e3e39791914a/)
Softmax weights with and without embedding normalization by $\sqrt{d_{k}}$.

**Step 3. Weighted Representation**  

Once normalized weights are obtained, the weighted representation for token $x_i$ is:  

$$
head_i = \sum_{j \leq i} \alpha_{ij} v_j
$$  

Here, $head_i$ represents the output of a **single attention head** (case $h=1$).  


**Step 4. Projection Back into Embedding Space ** 

Finally, the representation is projected back into the model’s embedding space:  

$$
o_i = head_i W^O
$$  

This projection serves two purposes:  

1. **Dimension alignment:**  
   - Vectors $q, k$ have dimension $[1, d_k]$.  
   - Vector $v$ has dimension $[1, d_v]$.  
   - After the weighted sum, $head_i$ has dimension $[1, d_v]$.  
   - To align with the model’s embedding dimension $d_{model}$, we apply the projection $W^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$.  

2. **Multi-head compatibility:**  
   - In the general case ($h > 1$), multiple heads are concatenated before being projected back with $W^O$.  
   - In this simplified case, $h = 1$.  


## **Attention: Multi-Head Extension** 

Once attention is formalized for a single head, the transition to **multi-head attention** involves two modifications:  

1. Adding multiple heads.  
2. Expanding the projection matrix $W^O$.  

Using multi-head attention allows the model to capture different types of dependencies in each head,  
and then aggregate them into a unified representation.  


**Formalization** 

Now, let each token vector $x_i$ be represented by three projections in each head $c$ out of $h$ heads:  

- $q^c_i = x_i W^Q_c$, dimension $[1, d_k]$  
- $k^c_i = x_i W^K_c$, dimension $[1, d_k]$  
- $v^c_i = x_i W^V_c$, dimension $[1, d_v]$  

**Step 1. Similarity Scores**  

For each previous token $x_j$, define the similarity to the current token $x_i$ in head $c$ as:  

$$
sim^c_{score}(x_i, x_j) = \frac{\langle q^c_i, k^c_j \rangle}{\sqrt{d_k}}
$$  

**Step 2. Normalization** 

Normalize these scores using softmax to obtain attention weights per head:  

$$
\alpha^c_{ij} = \text{softmax}(sim^c_{score}(x_i, x_j)), \quad \forall j \leq i
$$  

**Step 3. Weighted Representation** 

With normalized weights, the weighted representation for token $x_i$ in head $c$ is:  

$$
head^c_i = \sum_{j \leq i} \alpha^c_{ij} v^c_j
$$  

**Step 4. Projection Back into Embedding Space** 

The outputs of all heads are concatenated and projected back into the model’s embedding space:  

$$
o_i = \text{concat}[head^1_i, head^2_i, \dots, head^h_i] W^O
$$  

where the projection matrix $W^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$  
is expanded to accommodate all $h$ heads simultaneously.  


## **Circuits in Attention**  

The attention mechanism described in previous steps can be decomposed into two distinct subcircuits:  

- **QK-circuit (Query–Key circuit)** — decides *where to look*.  
- **OV-circuit (Output–Value circuit)** — decides *what to copy and how to integrate*.  


### **QK-circuit (Query–Key)**  

The QK-circuit is responsible for computing **attention logits (scores)**.  
It consists of dot products between query and key vectors:  

$$
sim_{score}(x_i, x_j) = \frac{\langle q_i, k_j \rangle}{\sqrt{d_k}}
$$  

This subcircuit determines how much the current token $x_i$ (via $q_i$) “aligns” with a past token $x_j$ (via $k_j$).  


### **OV-circuit (Output–Value)** 

The OV-circuit is responsible for **transferring information** from past tokens into the current representation:  

$$
o_i = head_i W^O = \sum_{j \leq i} \alpha_{ij} v_j W^O
$$  

where the weights $\alpha_{ij}$ are provided by the QK-circuit.  
This subcircuit transforms the value vectors (contents of past tokens) into a useful representation and integrates them into the current state.  


**Summary** 

- The **QK-circuit** defines the *structure* of the attention map.  
- The **OV-circuit** defines the *flow of information*.  

These circuits are fundamental analytical tools in mechanistic interpretability,  
and they frequently appear in research analyzing Transformer models.  

## **Transformer Block — LayerNorm and FFN**

The computation of self-attention, described earlier, forms the foundation of the **Transformer block**.  
In the literature, a Transformer block typically includes not only the self-attention layer but also three additional components:  

- **Feedforward layer (FFN)** — a fully connected layer applied after attention,  
- **Residual connections** — providing shortcut pathways that preserve information through the residual stream,  
- **Normalization layers** — most commonly **Layer Normalization (LayerNorm)**.  

Next I will examine these components individually,  
and then combine them to illustrate how they are integrated in the classical Transformer architecture.  

## **Feedforward Layer (FFN)** 

The **Feedforward Layer (FFN)** is a two-layer fully connected neural network.  
Its role is to transform each token representation in a **nonlinear way** and enrich it with new features.  

The FFN is defined as:  

$$
FFN(x) = \sigma(x W_1 + b_1) W_2 + b_2,
$$  

where:  

- $W_1 \in \mathbb{R}^{d \times d_{ff}}, \; b_1 \in \mathbb{R}^{d_{ff}}$  
- $W_2 \in \mathbb{R}^{d_{ff} \times d}, \; b_2 \in \mathbb{R}^{d}$  
- $\sigma(\cdot)$ is a nonlinear activation function (ReLU in the original Transformer).  

**Key FFN Properties**

- **Shared weights across positions.**  
  The same matrices $W_1, W_2$ are applied independently to each token,  
  ensuring positional consistency.  

- **Layer-specific parameters.**  
  Each Transformer block has its own set of FFN parameters.  

- **Increased hidden dimensionality.**  
  Typically, $d_{ff} \gg d$.  
  In the original Transformer: $d = 512$, $d_{ff} = 2048$.  
  This expansion allows the model to project a token into a richer feature space and then compress it back to dimension $d$.  

## **Layer Normalization (LayerNorm)**

**LayerNorm** is a step in the Transformer responsible for normalizing token embeddings.  
Its purpose is to stabilize the distribution of activations, which simplifies training with gradient-based methods  
by mitigating exploding or vanishing gradients.  

LayerNorm is a variant of **z-normalization** known from statistics.  
The transformation is:  

$$
x' = \frac{x - \mu}{\sigma},
$$  

where  

- $\mu = \frac{1}{d} \sum_{i=1}^d x_i$ — the mean over the $d$ dimensions of the token embedding,  
- $\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2}$ — the standard deviation.  

Difference from Standard z-Normalization is follows.  In the Transformer’s version, LayerNorm introduces **two learnable parameters**:  

$$
LayerNorm(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$  

- $\gamma$ — a scaling parameter,  
- $\beta$ — a bias (shift) parameter.  

These parameters preserve the **expressiveness** of embeddings after normalization,  
ensuring that the process does not collapse all values into the same range.  

## **Residual Stream and Residual Connections**  

The transformations inside a Transformer block can be expressed as a step-by-step sequence.  
To introduce the concepts of **residual stream** and **residual connections**, consider the following.  

Let $x_i^{(l)} \in \mathbb{R}^d$ denote the vector representation of token $i$ at layer $l$.  

The sequence of representations  

$$
\{x_i^{(0)}, x_i^{(1)}, \dots, x_i^{(L)}\}
$$  

formed from the input embedding through the final layer output,  
is called the **residual stream** of token $i$.  

Each representation is updated as follows:  

1. $t_i^1 = LayerNorm(x_i)$  
2. $t_i^2 = MultiHeadAttention(t_i^1, [t_1^1, \dots, t_N^1])$  
3. $t_i^3 = t_i^2 + x_i$  
4. $t_i^4 = LayerNorm(t_i^3)$  
5. $t_i^5 = FFN(t_i^4)$  
6. $h_i = t_i^5 + t_i^3$  

![transformer_block](https://ucarecdn.com/c9723058-68d7-417a-9bde-ad8baab6fd87/-/crop/1592x1074/125,0/-/preview/)

Step 3 highlights a key operation:  

$$
t_i^3 = t_i^2 + x_i
$$  

Let $t_i^3 = r_i$.  
This representation, formed as the **sum of the original vector $x_i$ and the attention output $t_i^2$**,  
is called a **residual connection**.  

Residual connections ensure that the original signal $x_i$ is preserved,  
while an additional correction term $t_i^2$ (the attention output) is added on top.  

Residual streams and residual connections describe the **flow of information** in a Transformer.  
They are therefore fundamental objects of study in mechanistic interpretability research.  

