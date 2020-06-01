from .infoGAN_3d import Generator, Discriminator, Plane
from .infoGAN_2d import Generator2D, Discriminator2D, Plane2D
from .infoGAN_crop import cropGenerator, cropDiscriminator, Plane
from .infoGAN_resnet import ResGenerator, ResDiscriminator
from .infoGAN_2d_resnet import ResGenerator2D, ResDiscriminator2D
from .infoGAN_bigi import bigiGenerator, bigiDiscriminator
#from .DCGAN import Generator, Discriminator
all = [ResGenerator2D, ResDiscriminator2D, Generator2D, Discriminator2D, Plane2D, bigiGenerator, bigiDiscriminator, cropGenerator,
       cropDiscriminator, ResGenerator, ResDiscriminator, Generator, Discriminator, Plane]
