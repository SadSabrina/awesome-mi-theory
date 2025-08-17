# **Basic Definitions**

Here, i will introduce the basic definitions used in Mechanistic Interpretability.  
To do so, we will consider a simplified neural network model:  

- A neural network $Net(X)$ that takes input data $X$.  
- Neurons of the network $u_{i,l}$ with weights $w_{i,j}^{(l)}$, where $i$ is the index of the neuron in layer $l$, and $j$ indexes its connections.  

![simple_net](https://ucarecdn.com/ee10b62f-b118-486b-8caf-1da93d1ae9a8/)

## **Features**

The primary goal of $MI$ is to understand which **features** a model encodes.  

**Definition.**  
A feature $f$ is the smallest stable unit of representation in a network $Net(X)$ that encodes properties of the input data within the activation space. The set of features forms the basis of internal concepts used by the model.  

Depending on perspective:  

- Features can be **human-interpretable** (*human-centric approach*).  
  For example, "cat" instead of a set of low-level properties such as *mammal + human’s friend + meow sound*.  

- Features can be **non-human-interpretable** (*non-human-centric approach*), yet still serve as essential elements of the basis of internal representation.  

In research practice, MI more often uses the second approach (treating features as basis units of knowledge), while in applied settings the emphasis is typically placed on human-interpretable features.  

### **Representation of Features**

There are several ways to describe how a feature can emerge in a network. The most common are:  

1. **As a set of pairs $\{u_{i,l}, w_{i,j}\}$** — neurons and their weights,  
   where the presence or absence of a feature is indicated by the strength of its activation on a given input $x^l$ passing through layer $l$.  

![gemma_features](https://ucarecdn.com/b2407c01-e593-4a48-a671-831beeb0268d/)

   *Example: Activation of neuron #44 in layer 0 of Gemma2-2b, Neuronpedia.*  

2. **As a direction in the activation space.**  
![linear direction](https://ucarecdn.com/a6ddd396-36e9-4125-bf96-ab29b0072717/)
3. **As a combination of layers** (for example, convolutions in a CNN).  

![cnn_example](https://ucarecdn.com/0ac38c23-6b7e-4994-9d3e-159e5e73e677/)

   *Example: Input transformation in a CNN, [[src]](https://becominghuman.ai/six-types-of-neural-networks-you-need-to-know-about-9a5e7604018c)*  


Early works often described features as **neurons**, and this representation is still used today.  
An important discovery from research mapping neurons $\to$ features is that:  

- A single neuron does not necessarily encode a single pattern.  
- Neurons that encode different patterns depending on the input are called **polysemantic neurons**.  
- Neurons that strictly encode a single pattern (e.g., a neuron detecting only curves) are called **monosemantic neurons**.  

**Example of a polysemantic neuron:**  
A neuron that activates both for bread (highlighting the edges of a loaf) and for bananas (highlighting the spots on the yellow peel).  

# Circuits  

A **circuit** $C$ is a stable structure of connections inside a network $Net(X)$, representing a subgraph of the computational graph that describes the interaction of features $f_i$ and implements a specific computation.  

When we speak of *stability*, we mean that a circuit has the following properties:  

- **Input independence.**  
  A circuit manifests not only on a single example but across a wide class of inputs $X$.  
  For example, the QK-circuit in a transformer, which implements the “token matching” mechanism, works in many contexts rather than just one sentence.  

- **Stability under parameter changes.**  
  Even if the weights of the network change slightly (e.g., during fine-tuning), the feature connections that form a circuit preserve their function.  
  This distinguishes circuits from random activation paths.  

- **Localizability in the computational graph.**  
  A circuit can be identified as a subgraph: a subset of neurons/weights/layers that reproducibly carries out a computation.  
  This means that if this subgraph is “cut out” or “masked,” the model’s behavior changes specifically in the way that the circuit is responsible for.  

If a **feature** explains *what* is encoded, a **circuit** explains *how* it is used and transformed.  

![circuits_vs_feature](https://ucarecdn.com/9989e635-1305-4c7e-9f44-26cea79ac947/)

A *feature* can be viewed as an elementary unit of knowledge, while a *circuit* is a combination of such units, organized through the weights and layers of the network, that enables the transformation of information into a specific component.  