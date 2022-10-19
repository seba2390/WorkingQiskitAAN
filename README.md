# MNIST handwritten digits in a "Quantum Circuit Associative Adversarial Network" framework. 
This repository implements an Associative Adversarial Network (AAN) to generate examples of the well-known MNIST handwritten digits. 
The network implements a Quantum Circuit Born Machine (QCBM) to learn the distribution of a layer of the Discriminator, of same size as the latent space, at each epoch. Samples of the learned quantum circuit (QC) is then used as an input for the Generator in the classical GAN constellation. <br>

### Main points.
- The optimal parameters of the QC was determined at each epoch using the gradient-free optimizing strategy known as 'COBYLA'.
- The optimal parameters of the Discriminator and the Generator was determined using the well-known 'Adam' optimizer.

The general strategy of this work is inspired by [[1]](#1), and details regarding the structure of the quantum circuit can therefore also be found in their paper, avalaible on arXiv [Generation of High-Resolution Handwritten Digits with an Ion-Trap Quantum Computer](https://arxiv.org/pdf/2012.03924.pdf).<br>



## References
<a id="1">[1]</a> 
Rudolph, M. S. et. al. (2022). 
Generation of High-Resolution Handwritten Digits with an Ion-Trap Quantum Computer. 
Phys. Rev. X 12.
