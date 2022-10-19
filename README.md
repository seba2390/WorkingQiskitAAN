# MNIST handwritten digits in a "Quantum Circuit Associative Adversarial Network" framework. 
This repository implements an Associative Adversarial Network (AAN) to generate examples of the well-known MNIST handwritten digits. 
The network implements a Quantum Circuit Born Machine (QCBM) to learn the distribution of a layer of the Discriminator, of same size as the latent space, at each epoch. Samples of the learned quantum circuit is then used as an input for the Generator in the classical GAN constellation. <br>


The main points of this work is inspired by [[1]](#1). <br>



## References
<a id="1">[1]</a> 
Rudolph, M. S. et. al. (2022). 
Generation of High-Resolution Handwritten Digits with an Ion-Trap Quantum Computer. 
Phys. Rev. X 12.
