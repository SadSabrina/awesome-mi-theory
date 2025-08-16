# Introduction to Mechanistic interpretability

## **A Bit of History**

Mechanistic Interpretability (MI) is a relatively new paradigm within the field of Explainable AI. However, its core idea is not new — it can be found in discussions within neuroscience [[1](https://www.sciencedirect.com/science/article/abs/pii/S1001074215000200), [2](https://link.springer.com/chapter/10.1007/978-3-319-22084-0_2), [3](https://www.sciencedirect.com/science/article/pii/S0888754314001773)] and cognitive science.  

![MI_intro_IMG](https://ucarecdn.com/d2c11722-9a16-42d4-9145-ee8f52942cf9/-/crop/1132x369/0,3/-/preview/)
*Philosophy of Cognitive Science in the Age of Deep Learning, May 2024*  

When applied to models, the concept of mechanistic interpretability can be used not only for neural networks but also for other algorithms, such as Support Vector Machines (SVMs) [[4](https://link.springer.com/chapter/10.1007/978-81-322-1602-5_70)]. This highlights that mechanistic interpretability is not a unique discovery, but rather an intriguing and multifaceted idea that has proven useful in analyzing complex models, including deep neural networks.  

## **From CNNs to Transformers**  

The concept of *mechanistic interpretability* first appeared in public discussions in the context of Convolutional Neural Networks (CNNs), specifically in the paper [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/). This work explored how the synergy between activations and weights across different layers could be linked to the recognition of specific objects.  

![zoom_in_intro_to_circuits](https://ucarecdn.com/2de526c4-c1fb-4fc7-907b-b1f734dd08f2/)
Later, interest in mechanistic interpretability gradually shifted toward transformer architectures, as highlighted by various discussions and studies in the field. Researchers began to ask:  

> *"Can we redesign transformer-based language models into computer programs that are understandable to humans?"*  

This growing curiosity led to a surge in the popularity of mechanistic interpretability, which has since become closely tied to the transformer architecture. However, its origins can be traced back to CNNs.  

## **Definition and Distinction of Mechanistic Approaches**  

The overarching goal of Mechanistic Interpretability (MI) — much like traditional XAI — is to describe and explain the internal mechanisms of artificial intelligence models.  

A possible formal definition could be:  

    *MI refers to any research aimed at describing the internal structures and mechanisms of a model.*

In other words, MI always involves analyzing the internal representations and components, striving to understand how individual elements (neurons) and their interactions shape the overall behavior of the network.  

Another way to highlight this distinction is through the categorization of XAI methods into **bottom-up** and **top-down** approaches:  

- **Bottom-up approaches** focus on smaller units (individual neurons, activations, parameters). MI falls into this category.  
- **Top-down approaches** focus on high-level model behavior or larger architectural blocks. Sometimes, but not always, top-down analysis can be connected back to finer mechanisms.  

In practice, the boundaries are blurred: both strategies may eventually include the study of neurons and representations inside models. For example, one might identify a specific capability of a model (such as solving a given task) through probing, and then zoom in on the layers, modules, or groups of neurons responsible for it.  
 
![three_steps_MI](https://ucarecdn.com/688f59f7-3d81-4f5e-9643-6eee85c7153e/)
One of the possible decompositions of MI into three steps, [source](https://www.alignmentforum.org/posts/64MizJXzyvrYpeKqm/sparsify-a-mechanistic-interpretability-research-agenda)