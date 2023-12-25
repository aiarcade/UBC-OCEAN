from pl_bolts.models.gans import DCGAN
import torch
import torchvision.transforms as T

batch_size=1
latent_dim=100
noise = torch.rand(batch_size, latent_dim)
gan = DCGAN.load_from_checkpoint('../out/GAN_CC.ckpt')
img = gan(noise).squeeze(0)

transform = T.ToPILImage()

img = transform(img)

img.save("out.png")