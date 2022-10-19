# MNIST handwritten digits in a "Quantum Circuit Associative Adversarial Network" framework. 
This repository implements an Associative Adversarial Network (AAN) to generate examples of the well-known MNIST handwritten digits. 
The network implements a Quantum Circuit Born Machine (QCBM) to learn the distribution of a layer of the Discriminator, of same size as the latent space, at each epoch. Samples of the learned quantum circuit (QC) is then used as an input for the Generator in the classical GAN constellation. <br>

### Main points.
- The Quantum circuit is simulated using the 'Aer-simulator' in Qiskits framework.
- The optimal parameters of the QC was determined at each epoch using the gradient-free optimizing strategy known as 'COBYLA'.
- The Discriminator and the Generator networks are structured approximatly as each others inverses, and generally follows the standard approach of DCGAN as presented by [[1]](#1).
- The optimal parameters of the Discriminator and the Generator was determined using the well-known 'Adam' optimizer using PyTorch.
- The classical part of the GAN is written such that both forward- and backwardspass can be done on the CPU, or one the GPU (using CUDA), if available.

#### Generator progress for digits (3,7,9):
![alt text](https://github.com/seba2390/WorkingQiskitAAN/blob/main/media/379.gif "Logo Title Text 1")

The general strategy of this work is inspired by [[2]](#2), and details regarding the structure of the quantum circuit can therefore also be found in their paper, avalaible on arXiv [Generation of High-Resolution Handwritten Digits with an Ion-Trap Quantum Computer](https://arxiv.org/pdf/2012.03924.pdf).<br>

## References
<a id="1">[1]</a> 
Radford, A. et. al. (2015). 
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. 
arXiv:1511.06434.

<a id="2">[2]</a> 
Rudolph, M. S. et. al. (2022). 
Generation of High-Resolution Handwritten Digits with an Ion-Trap Quantum Computer. 
Phys. Rev. X 12.
