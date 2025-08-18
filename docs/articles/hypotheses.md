# Four (?) Key Hypotheses of Mechanistic Interpretability  

To better navigate the directions explored through MI methods, it is useful to establish the **key hypotheses** in the field.  
The number of such hypotheses varies across different reviews — ranging from a single central hypothesis (most often the *universality hypothesis*) to several.  

Here, I outline **four main hypotheses**, which together form a kind of foundation for MI research directions.  

## **Hypothesis 1: Superposition**

**Formulation:**  
A neural network with $N$ neurons can represent a number of features $K$ such that $K \gg N$.  

In other words, the number of features in a network is greater than the number of neurons.  
This hypothesis suggests that the network "packs" information into different directions of a shared space,  
and that each discovered feature may influence other features when it is modified.  

Under this assumption, each neuron or direction can encode either a single feature or multiple ones.  
This leads to the following definitions:  

- **Monosemantic neuron (feature):** a neuron (feature) specialized in a single concept.  
- **Polysemantic neuron (feature):** a neuron (feature) encoding multiple concepts simultaneously.  

Hypothetically, monosemantic features are desirable, since they enable precise control over a model.  
For example, one could disable features responsible for harmful behavior,  
with the confidence that this intervention would not affect unrelated tasks.  

![superposition](https://ucarecdn.com/c5ddd512-9d0a-4a22-ad07-f545b93fda94/)
[Source](https://arxiv.org/pdf/2404.14082)  

![Polysemantic](https://ucarecdn.com/4993b5f3-0a25-4662-b8ed-f04580243810/)
*Polysemantic neuron Example: A neuron encoding both visual patterns of cats and cars.*  

## **Hypothesis 2: Linear Representation** 

The **Linear Representation Hypothesis** manifests itself in three main ways.  

**Formulation:**  
Let $h \in \mathbb{R}^d$ be the hidden state of a neural network, and let $f_i$ be a feature.  
Each feature $f_i$ corresponds to a linear functional or direction $w_c \in \mathbb{R}^d$, such that:  

### 1. Subspace  
A feature $f_i$ can be described by a one-dimensional subspace. For example, representations:  

- $r_1 = \text{vec}(\text{woman}) - \text{vec}(\text{man})$  
- $r_2 = \text{vec}(\text{queen}) - \text{vec}(\text{king})$  

and similar $r_i$ all belong to the same subspace describing the concept of *gender (male/female)*.  

### 2. Measurement  
There exists a linear probe that approximates the probability of a feature being present:  

$$
P(f_i \mid h) \approx \sigma(\langle w_{f_i}, h \rangle + b_{f_i})
$$  

where $w_{f_i}$ is the direction corresponding to the presence of feature $f_i$.  

### 3. Intervention (Steering)  
A feature $f_i$ in the hidden state $h$ can be transformed into another feature $f_j$  
by adding its direction $w_{f_j}$:  

$$
h' = h + \alpha w_{f_j}
$$  

where $\alpha \in \mathbb{R}$ controls the intensity of the feature.  


In general, this hypothesis states that features $f_i$ can be expressed or approximated using linear functions.  

![linear_repr](https://ucarecdn.com/a207cb2b-979f-443f-8ae8-087884b95761/)
[The Linear Representation Hypothesis and the Geometry of Large Language Models](https://arxiv.org/pdf/2311.03658)


### Exceptions  

There are tasks where networks form **nonlinear representations**.  
One example is modular arithmetic (e.g., working with dates), where features cannot be separated by a linear hyperplane.  

*Example from GPT-2 small:*  

![non_linear](https://ucarecdn.com/7a983b42-3f36-4b1e-9044-7a030630ab76/)

*"[Not all LM features are linear.](https://arxiv.org/pdf/2405.14860)"*  

## Hypothesis 3: Universality  

**Formulation:**  

Let $Net_1: \mathcal{X}_1 \to \mathbb{R}^d$ and $Net_2: \mathcal{X}_2 \to \mathbb{R}^d$ be two networks trained on tasks $T_1, T_2$ (which may be the same or different).  
Let $F^{(1)} = \{ f^{(1)}_i \}$ and $F^{(2)} = \{ f^{(2)}_j \}$ be the sets of features learned by the respective networks.  

The **Universality Hypothesis** states that there exist subsets $F^{(1)*} \subseteq F^{(1)}$, $F^{(2)*} \subseteq F^{(2)}$, for which there is a bijection  

$$
\pi : F^{(1)*} \to F^{(2)*}
$$  

In other words, universal features and circuits are the intersection of feature sets that reappear across different networks (regardless of architecture, initialization, or even task), and between which there exists a reversible transformation.  

This hypothesis is based on the observation that models tend to learn **repeating patterns** that appear across different models and tasks.  
Researchers have called these recurring patterns **motifs**.  

- A simple example of a motif is **curve detectors** discovered in vision models.  
- More complex motifs include **subnetworks (circuits)** specializing in particular tasks — for example, **induction heads** in transformers.  

![univers](https://ucarecdn.com/38e7133d-45b5-4328-927a-ec11990f4b2b)
[*Universality in CNNs*](https://distill.pub/2020/circuits/zoom-in/)


### Significance of Universality [src](https://distill.pub/2020/circuits/zoom-in/#claim-3) 

Testing the universality hypothesis has major implications:  

- If the hypothesis holds strictly — i.e., there exists a fixed and interpretable set of features —  
  one could imagine a sort of **“periodic table of visual features”**, analogous to Mendeleev’s table of chemical elements.  

- On the other hand, if the hypothesis is largely false, then research must focus on a few models of special societal importance —  
  with the hope that these models stop changing every year. 


## Hypothesis 4: Internal World Model  

**Formulation:**  

As neural networks grow in scale and complexity, they begin to form an **internal world model** — a structure that reflects the causal relationships underlying the training data.  
Such models go beyond mere memorization of surface-level statistics and instead reproduce the dynamics of the environment and the world, making them similar to **simulators**.  

This interpretation helps explain phenomena such as:  

- **Hallucinations** — the model “fills in” missing details based on its internal simulation.  
- **Grokking** — the sudden emergence of a capability when the internal world structure stabilizes at some point.  


### Example: Simulation of Agents  

Large language models can imitate reasoning and agent-like behavior with goals:  
they can conduct dialogue in the voice of fictional characters, reconstruct hidden states (such as intentions or plans), and simulate interactions.  

This suggests that they should be seen not only as predictive text models but also as systems capable of building internal scenarios.  
Testing this hypothesis is particularly important in the context of **AI Safety** — whether there exists a genuine risk arising from a model’s internal world representation.  

![parrot](https://ucarecdn.com/900cde24-4990-4c8a-80db-95be0d6155bc/-/crop/1589x1079/178,0/-/preview/)

### Criticism  

There is also skepticism toward this hypothesis — often expressed as the [*“stochastic parrot”*](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922) theory:  
models simply reproduce probabilistic token sequences without possessing any actual world model.  

The debate remains open as to whether LLMs are true simulators of the world or merely advanced statistical machines.  



