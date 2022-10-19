import numpy as np
import torch
import torchvision
from tqdm import tqdm
from Discriminator import Discriminator
from Generator import Generator
import matplotlib.pyplot as plt
import wandb
import os
import scipy
from ansatz import *
from cost import *


class GenerativeAdversarialNetwork(torch.nn.Module):
    def __init__(self, latent_size: int = 35):
        super(GenerativeAdversarialNetwork, self).__init__()

        # torch.manual_seed(0)  # For reproducibility

        self.latent_dims = latent_size
        self.image_dims = (28, 28)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print("Using device: ", self.device)

        self.discriminator = Discriminator(image_dims=self.image_dims, latent_dim=self.latent_dims)
        self.generator = Generator(image_dims=self.image_dims, latent_dims=self.latent_dims)
        self.discriminator.to(self.device)
        self.generator.to(self.device)

        self._N_QUBITS = self.latent_dims
        self._L_DEPTH = 2
        self._NR_SHOTS = 2000
        self._ALL_2_ALL = True
        self._USE_ROWS = False
        self._I_MAX = 5
        if self._ALL_2_ALL:
            _NR_PARAMS = int(self._L_DEPTH * (2 * self._N_QUBITS + self._N_QUBITS * (self._N_QUBITS - 1) / 2))
        else:
            _NR_PARAMS = int(self._L_DEPTH * (2 * self._N_QUBITS + self._N_QUBITS - 1))
        self._THETA_INIT = torch.randn(size=(_NR_PARAMS,)).tolist()

    # To save images in grid layout
    @staticmethod
    def save_image_grid(epoch: int, images: torch.Tensor, ncol: int):
        image_grid = torchvision.utils.make_grid(images, ncol)  # Images in a grid
        image_grid = image_grid.permute(1, 2, 0)  # Move channel last
        image_grid = image_grid.cpu().numpy()  # To Numpy

        plt.imshow(image_grid)
        plt.xticks([])  # To make ticks invisible
        plt.yticks([])  # To make ticks invisible
        plt.savefig(f'progress_pics/image_{epoch}.jpg')
        plt.close()

    def train_network(self, dataloader, lrs: tuple[float, float] = (0.001, 0.001),
                      wds: tuple[float, float] = (0.000, 0.000), epochs: int = 100,
                      betas: tuple[tuple[float, float], tuple[float, float]] = ((0.9, 0.999), (0.9, 0.999)),
                      save_images: bool = False, use_wandb: bool = False, label_smooth: float = 0.0):

        # Real and fake labels
        real_targets = torch.ones(dataloader.batch_size, 1, device=self.device) * (1 - label_smooth)
        fake_targets = torch.zeros(dataloader.batch_size, 1, device=self.device) + label_smooth

        # Optimizers
        """Tested working optimizers: Adam, RMSprop, RAdam, NAdam"""
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lrs[0],
                                       weight_decay=wds[0], betas=betas[0])
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lrs[1],
                                       weight_decay=wds[1], betas=betas[1])

        # Training loop
        disc_losses, gen_losses = [], []
        for epoch in tqdm(range(epochs)):
            d_losses = []
            g_losses = []

            weights = torch.bernoulli(torch.abs(self.discriminator.fc1[0].weight))
            my_cost = get_cost(X_train=weights,
                               _nr_qubits=self._N_QUBITS,
                               _layer_depth=self._L_DEPTH,
                               _shots=self._NR_SHOTS,
                               _all_2_all=self._ALL_2_ALL,
                               _use_rows=self._USE_ROWS)

            my_result = scipy.optimize.minimize(fun=my_cost, x0=self._THETA_INIT,
                                                method="COBYLA", options={'maxiter': self._I_MAX})
            self._THETA_INIT = my_result['x']

            for images, labels in dataloader:
                images = images.to(self.device)
                # ================================ #
                #  Discriminator Network Training  #
                # ================================ #

                # Loss with MNIST image inputs and real_targets as labels
                self.discriminator.train()
                real_d_predictions = self.discriminator.forward(images)
                d_loss = self.discriminator.loss(targets=real_targets, predictions=real_d_predictions)

                # Generate images in eval mode
                self.generator.eval()
                with torch.no_grad():
                    latent_vectors = sample_qcirc(params=self._THETA_INIT,
                                                  _nr_samples=dataloader.batch_size,
                                                  _nr_qubits=self.latent_dims,
                                                  _layer_depth=self._L_DEPTH,
                                                  _all_2_all=True,
                                                  _uniform_warm_start=True).to(self.device)
                    generated_images = self.generator.forward(latent_vectors)

                # Loss with generated image inputs and fake_targets as labels
                fake_d_predictions = self.discriminator.forward(generated_images)
                d_loss += self.discriminator.loss(targets=fake_targets, predictions=fake_d_predictions)

                # Optimizer updates the discriminator parameters
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                # ===============================#
                #  Generator Network Training   #
                # ===============================#

                # Generate images in train mode
                self.generator.train()
                latent_vectors = sample_qcirc(params=self._THETA_INIT,
                                              _nr_samples=dataloader.batch_size,
                                              _nr_qubits=self.latent_dims,
                                              _layer_depth=self._L_DEPTH,
                                              _all_2_all=True,
                                              _uniform_warm_start=True).to(self.device)
                generated_images = self.generator.forward(latent_vectors)

                # Loss with generated image inputs and real_targets as labels
                self.discriminator.eval()  # eval but we still need gradients
                fake_d_predictions_2 = self.discriminator.forward(generated_images)
                g_loss = self.discriminator.loss(targets=real_targets, predictions=fake_d_predictions_2)

                # Optimizer updates the generator parameters
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                # Keep losses for logging
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            with torch.no_grad():
                avg_d_loss = torch.mean(torch.tensor(d_losses)).item()
                avg_g_loss = torch.mean(torch.tensor(g_losses)).item()
                if avg_g_loss < 0.005 or avg_d_loss < 0.005:
                    print(f"Discriminator loss: {avg_d_loss}, or Generator loss: {avg_g_loss} is to low - breaking.")
                    wandb.alert(
                        title="Low accuracy",
                        text=f"Discriminator loss: {avg_d_loss}, or Generator loss: {avg_g_loss} is to low.",
                        level=wandb.AlertLevel.WARN
                    )
                    break
                disc_losses.append(avg_d_loss)
                gen_losses.append(avg_g_loss)
                if use_wandb:
                    wandb.log({"avg. discriminator loss": avg_d_loss,
                               "avg. generator loss": avg_g_loss})

            # Save images and log to wandb
            with torch.no_grad():
                nr_images = 4  # Choose a number w. integer square root
                if save_images:
                    if (epoch + 1) % 1 == 0:
                        self.generator.eval()

                        latent_vectors = sample_qcirc(params=self._THETA_INIT,
                                                      _nr_samples=dataloader.batch_size,
                                                      _nr_qubits=self.latent_dims,
                                                      _layer_depth=self._L_DEPTH,
                                                      _all_2_all=True,
                                                      _uniform_warm_start=True).to(self.device)

                        generated_image_arrays = self.generator.forward(latent_vectors)
                        self.save_image_grid(epoch, generated_image_arrays, ncol=int(np.sqrt(nr_images)))
                        if use_wandb:
                            wandb.log({"Generator examples": wandb.Image(f'progress_pics/image_{epoch}.jpg')})
                else:
                    if use_wandb:
                        if (epoch + 1) % 1 == 0:
                            self.generator.eval()
                            latent_vectors = sample_qcirc(params=self._THETA_INIT,
                                                          _nr_samples=dataloader.batch_size,
                                                          _nr_qubits=self.latent_dims,
                                                          _layer_depth=self._L_DEPTH,
                                                          _all_2_all=True,
                                                          _uniform_warm_start=True).to(self.device)
                            generated_image_arrays = self.generator.forward(latent_vectors)
                            self.save_image_grid(epoch, generated_image_arrays,
                                                 ncol=int(np.sqrt(nr_images)))
                            wandb.log({"Generator examples": wandb.Image(f'progress_pics/image_{epoch}.jpg')})
                            os.system("cd progress_pics && rm *.jpg")  # Deleting picture as 'save_images' is False.

        return disc_losses, gen_losses
