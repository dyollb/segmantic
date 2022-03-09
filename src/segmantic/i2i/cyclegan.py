import itertools

import pytorch_lightning as pl
import torch

from .networks import GANLoss, NLayerDiscriminator, ResnetGenerator, get_norm_layer
from .util.image_pool import ImagePool


def make_generator(spatial_dims: int, norm_type: str, n_blocks: int) -> ResnetGenerator:
    norm_layer = get_norm_layer(spatial_dims, norm_type)
    return ResnetGenerator(
        spatial_dims,
        input_nc=1,
        output_nc=1,
        ngf=64,
        norm_layer=norm_layer,
        use_dropout=False,
        n_blocks=n_blocks,
    )


def make_discriminator(spatial_dims: int, norm_type: str) -> NLayerDiscriminator:
    norm_layer = get_norm_layer(spatial_dims, norm_type)
    return NLayerDiscriminator(
        spatial_dims=spatial_dims, input_nc=1, ndf=64, n_layers=3, norm_layer=norm_layer
    )


class CycleGANModel(pl.LightningModule):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(
        self,
        lambda_A: float,
        lambda_B: float,
        lambda_identity: float,
        lr: float,
        beta1: float,
    ):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(CycleGANModel, self).__init__()
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_identity = lambda_identity

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [
            "D_A",
            "G_A",
            "cycle_A",
            "idt_A",
            "D_B",
            "G_B",
            "cycle_B",
            "idt_B",
        ]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if (
            lambda_identity > 0.0
        ):  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = (
            visual_names_A + visual_names_B
        )  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ["G_A", "G_B", "D_A", "D_B"]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = make_generator(spatial_dims=2, norm_type="instance", n_blocks=6)
        self.netG_B = make_generator(spatial_dims=2, norm_type="instance", n_blocks=6)

        self.netD_A = make_discriminator(spatial_dims=2, norm_type="instance")
        self.netD_B = make_discriminator(spatial_dims=2, norm_type="instance")

        # if (lambda_identity > 0.0):  # only works when input and output images have the same number of channels
        #    assert opt.input_nc == opt.output_nc

        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        # define loss functions
        self.criterionGAN = GANLoss("lsgan").to(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            lr=lr,
            betas=(beta1, 0.999),
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr=lr,
            betas=(beta1, 0.999),
        )
        # self.optimizers = [self.optimizer_G, self.optimizer_D]

        # self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input["A"].to(self.device)
        self.real_B = input["B"].to(self.device)
        # self.image_paths = input["A_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = (
                self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            )
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = (
                self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            )
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = (
            self.loss_G_A
            + self.loss_G_B
            + self.loss_cycle_A
            + self.loss_cycle_B
            + self.loss_idt_A
            + self.loss_idt_B
        )
        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(
            [self.netD_A, self.netD_B], False
        )  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate gradients for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
