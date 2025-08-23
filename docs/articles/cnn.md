# **Convolutional Neural Networks (CNNs)** 

**Convolutional Neural Networks (CNNs)** are an architecture with a rich history.  
They were originally developed for image recognition tasks and, for a long time, represented the dominant paradigm in computer vision.  
Although Transformers have largely displaced CNNs in recent years, CNNs remain an important object of study in **mechanistic interpretability (MI)**.  

CNNs are particularly useful because they provide:  

- **An intuitive structure for analysis** — convolutional kernels are explicitly designed to learn features.  
- **Empirical alignment with human intuition** — many discovered features (e.g., edge, texture, and shape detectors) can be directly interpreted.  
- **Traceability of feature formation** — from low-level (gradients, edges) to high-level (object parts, full objects) representations, given sufficient network depth.  

## **CNN Architecture**  

A convolutional neural network (CNN) typically consists of the following sequence of components:  

1. **Convolutional layers (Conv layers):**  
   Apply filters (convolution kernels) of size $k \times k$, producing feature maps.  

2. **Nonlinearities (ReLU and its variants, sigmoid):**  
   Introduce nonlinearity into the model and regulate the numerical range of activations.  

3. **Pooling layers:**  
   Reduce dimensionality (e.g., max pooling, average pooling), thereby increasing efficiency and spatial invariance.  

4. **Fully connected layers (FC):**  
   Act as a classifier on top of the extracted features.  


![cnn_func](https://ucarecdn.com/e2e36bfa-7226-4df4-8794-6f7a023fde80/)
Nonlinear Activation Functions in CNN, [source](https://arxiv.org/pdf/1609.04112)

![vgg16](https://ucarecdn.com/6d815a74-2a65-49a0-a33d-fc3aeafe80bf/)
An example of CNN, VGG-16, [src](https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f)

## **Convolution Operation in CNNs**  

The feature after applying a convolution is computed as:  

$$
f_{ijc} =
\sum_{p=1}^{C_{\text{in}}}
\sum_{k_1=-\lfloor k/2 \rfloor}^{\lfloor k/2 \rfloor}
\sum_{k_2=-\lfloor k/2 \rfloor}^{\lfloor k/2 \rfloor}
W^{c}_{\,\lfloor k/2 \rfloor + 1 + k_1,\;\lfloor k/2 \rfloor + 1 + k_2,\;p}
\cdot X_{\,i+k_1,\; j+k_2,\; p}
+ b_c
$$  

where:  

- $X$ — input feature map of size $H \times W \times C_{in}$  
  ($H, W$ — height and width; $C_{in}$ — number of input channels, e.g., 3 for RGB).  
- $W^c$ — convolution kernel (filter) associated with output channel $c$, of size $k \times k \times C_{in}$.  
- $b_c$ — bias term added to each element of the output feature map.  
- $f_{ijc}$ — value of the output feature map at position $(i, j)$ in channel $c$.  


**Summation Dimensions ** 

- Over $k_1, k_2$ — sliding across the $k \times k$ kernel window.  
- Over $p$ — combining contributions across all input channels.  

**Intuitive Computation of $f_{ijc}$**  

1. Take the local $k \times k$ window around position $(i, j)$ in the input $X$.  
2. Multiply this window elementwise by the kernel weights $W^c$.  
3. Sum the results across all channels $p$.  
4. Add the bias $b_c$.  


<details><summary><strong>Role of Nonlinear Activation Functions</strong></summary>

Nonlinearities are necessary because:  

- <strong>Breaking linearity.</strong>
  Convolution is a linear operation. A stack of purely convolutional layers, no matter how deep,  
  remains a linear model. Nonlinearities (e.g., ReLU, sigmoid) make the network expressive.  

- <strong>Improving stability and feature quality.</strong> 
  Nonlinear functions help prevent exploding activations, improve gradient descent convergence,  
  and suppress insignificant values. This allows the network to distinguish meaningful patterns from noise.  
</details>

**Geometric Interpretation**  

- Negative activation values are often treated as **insignificant**.  
- Intuitively, the larger the positive value after convolution, the stronger the similarity between the local input patch $x$ and the kernel $W$ (in terms of dot product).  
- Since a filter’s purpose is to detect specific features, a **large positive response** indicates that the corresponding feature is present.  


<details><summary><strong>Linearity Reminder</strong></summary>

An operator \(W\) is **linear** if it satisfies:  

1. <strong>Additivity (preserves sums):</strong>
   $$
   W(x_1 + x_2) = W(x_1) + W(x_2)
   $$  

2. <strong>Homogeneity (preserves scalar multiplication):</strong>
   $$
   W(\alpha x) = \alpha W(x), \quad \forall \alpha \in \mathbb{R}
   $$  

</details>

## **Nature of Features in CNNs**  

Since CNNs are built on the idea of convolutional kernels,  
during training they naturally learn to highlight meaningful components —  
those that are important according to the optimized loss function.  

A **convolution kernel** is a small $k \times k$ filter that slides across the image,  
computing local linear combinations of pixels.  


### **Hierarchical Feature Extraction**  

The combination of convolution kernels with parameter optimization  
gives rise to the **hierarchical organization of features**:  

- **Early convolutional layers**  
  learn simple, low-level features: gradients, edges, corners, fine details.  

- **Intermediate layers**  
  begin to capture more complex textures and shapes.  

- **Deeper convolutional layers**  
  learn high-level patterns: object parts and their combinations.  

- **Near the output (fully connected layers)**  
  the network forms representations that can be interpreted  
  as abstract features of entire objects.  

![googlenet](https://ucarecdn.com/f7136fa8-5882-40cf-a06f-2c83f2280561/-/crop/2807x865/0,31/-/preview/)

GoogLeNet, [src](https://distill.pub/2017/feature-visualization/)

This gradual progression of feature complexity is a **core property of CNNs**  
and makes them an intuitive starting point for studying feature analysis in **mechanistic interpretability**.  

## **Early Assumptions of Mechanistic Interpretability in CNNs**  

The first assumptions of **mechanistic interpretability** were demonstrated specifically in CNNs.  
They were formulated and illustrated with empirical examples.  
This introduction to the field consisted of three statements — not proven, but empirically observed.  
Such statements are often called *speculative*.  

In their raw form, they were as follows:  


**Statement 1. Features**  
**Features** are the fundamental units of neural networks.  
They can be studied and understood as relatively autonomous objects.  
In terms of linear algebra, they correspond to **directions in activation space**.  

**Statement 2. Circuits** 
Features are connected by weights, forming **circuits**.  
These circuits can be studied as interactions between features,  
from which more complex representations are built.  

**Statement 3. Universality** 
Analogous features and circuits emerge across different models and tasks.  
That is, there exists a degree of **universality** in which features are learned and how they are combined.  

![zoom_in_circuits](https://ucarecdn.com/ea18fad6-b665-4550-8e23-66501ebddc89/)
Source: [Zoom In: An Introduction to Circuit](https://distill.pub/2020/circuits/zoom-in/)

As we can see, what was missing at this stage was only the **Internal World Model Hypothesis**,  
which emerged later in the analysis of large models solving a broader range of tasks than CNNs.  
