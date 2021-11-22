import torch
import torch.nn as nn
import pytorch_lightning as pl

from collections import OrderedDict

from .networks import ResnetGenerator, NLayerDiscriminator
from .networks import get_norm_layer, GANLoss
from .util.image_pool import ImagePool


def make_generator(spatial_dims: int, norm_type: str, n_blocks: int):
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


def make_discriminator(spatial_dims: int, norm_type: str):
    norm_layer = get_norm_layer(spatial_dims, norm_type)
    return NLayerDiscriminator(
        spatial_dims=spatial_dims, input_nc=1, ndf=64, n_layers=3, norm_layer=norm_layer
    )


# TODO: check also: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html
class GAN(pl.LightningModule):
    def __init__(
        self,
        spatial_dims: int,
    ):
        super(GAN, self).__init__()
        self.save_hyperparameters()

        # networks
        self.G_AtoB = make_generator(
            spatial_dims, norm_type="instance", n_blocks=6  # TODO: check defaults
        )
        self.G_BtoA = make_generator(spatial_dims, norm_type="instance", n_blocks=6)
        self.D_A = make_discriminator(spatial_dims, norm_type="instance")
        self.D_B = make_discriminator(spatial_dims, norm_type="instance")

        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        # define loss functions
        self.criterionGAN = GANLoss("lsgan").to(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

    def forward(self, xa, xb):
        # TODO: missing cycle rec_A, and rec_B
        return self.G_BtoA(xb), self.G_AtoB(xa)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch

        # train generator
        if optimizer_idx == 0:

            # generate images
            fake_A, fake_B = self(real_A, real_B)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        # log sampled images
        pass


# https://www.kaggle.com/bootiu/cyclegan-pytorch-lightning


class CycleGAN_LightningSystem(pl.LightningModule):
    def __init__(self, spatial_dims, lr, transform, reconstr_w=10, id_w=2):
        super(CycleGAN_LightningSystem, self).__init__()
        self.G_basestyle = make_generator(
            spatial_dims, norm_type="instance", n_blocks=6  # TODO: check defaults
        )
        self.G_stylebase = make_generator(
            spatial_dims, norm_type="instance", n_blocks=6
        )
        self.D_base = make_discriminator(spatial_dims, norm_type="instance")
        self.D_style = make_discriminator(spatial_dims, norm_type="instance")
        self.lr = lr
        self.transform = transform
        self.reconstr_w = reconstr_w
        self.id_w = id_w
        self.cnt_train_step = 0
        self.step = 0

        self.mae = nn.L1Loss()
        self.generator_loss = nn.MSELoss()
        self.discriminator_loss = nn.MSELoss()
        self.losses = []
        self.G_mean_losses = []
        self.D_mean_losses = []
        self.validity = []
        self.reconstr = []
        self.identity = []

    def configure_optimizers(self):
        self.g_basestyle_optimizer = torch.optim.Adam(
            self.G_basestyle.parameters(), lr=self.lr["G"], betas=(0.5, 0.999)
        )
        self.g_stylebase_optimizer = torch.optim.Adam(
            self.G_stylebase.parameters(), lr=self.lr["G"], betas=(0.5, 0.999)
        )
        self.d_base_optimizer = torch.optim.Adam(
            self.D_base.parameters(), lr=self.lr["D"], betas=(0.5, 0.999)
        )
        self.d_style_optimizer = torch.optim.Adam(
            self.D_style.parameters(), lr=self.lr["D"], betas=(0.5, 0.999)
        )

        return [
            self.g_basestyle_optimizer,
            self.g_stylebase_optimizer,
            self.d_base_optimizer,
            self.d_style_optimizer,
        ], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        base_img, style_img = batch
        b = base_img.size()[0]

        valid = torch.ones(b, 1, 30, 30).cuda()
        fake = torch.zeros(b, 1, 30, 30).cuda()

        # Train Generator
        if optimizer_idx == 0 or optimizer_idx == 1:
            # Validity
            # MSELoss
            val_base = self.generator_loss(
                self.D_base(self.G_stylebase(style_img)), valid
            )
            val_style = self.generator_loss(
                self.D_style(self.G_basestyle(base_img)), valid
            )
            val_loss = (val_base + val_style) / 2

            # Reconstruction
            reconstr_base = self.mae(
                self.G_stylebase(self.G_basestyle(base_img)), base_img
            )
            reconstr_style = self.mae(
                self.G_basestyle(self.G_stylebase(style_img)), style_img
            )
            reconstr_loss = (reconstr_base + reconstr_style) / 2

            # Identity
            id_base = self.mae(self.G_stylebase(base_img), base_img)
            id_style = self.mae(self.G_basestyle(style_img), style_img)
            id_loss = (id_base + id_style) / 2

            # Loss Weight
            G_loss = val_loss + self.reconstr_w * reconstr_loss + self.id_w * id_loss

            return {
                "loss": G_loss,
                "validity": val_loss,
                "reconstr": reconstr_loss,
                "identity": id_loss,
            }

        # Train Discriminator
        elif optimizer_idx == 2 or optimizer_idx == 3:
            # MSELoss
            D_base_gen_loss = self.discriminator_loss(
                self.D_base(self.G_stylebase(style_img)), fake
            )
            D_style_gen_loss = self.discriminator_loss(
                self.D_style(self.G_basestyle(base_img)), fake
            )
            D_base_valid_loss = self.discriminator_loss(self.D_base(base_img), valid)
            D_style_valid_loss = self.discriminator_loss(self.D_style(style_img), valid)

            D_gen_loss = (D_base_gen_loss + D_style_gen_loss) / 2

            # Loss Weight
            D_loss = (D_gen_loss + D_base_valid_loss + D_style_valid_loss) / 3

            # Count up
            self.cnt_train_step += 1

            return {"loss": D_loss}

    def training_epoch_end(self, outputs):
        self.step += 1

        avg_loss = sum(
            [
                torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 4
                for i in range(4)
            ]
        )
        G_mean_loss = sum(
            [
                torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 2
                for i in [0, 1]
            ]
        )
        D_mean_loss = sum(
            [
                torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 2
                for i in [2, 3]
            ]
        )
        validity = sum(
            [
                torch.stack([x["validity"] for x in outputs[i]]).mean().item() / 2
                for i in [0, 1]
            ]
        )
        reconstr = sum(
            [
                torch.stack([x["reconstr"] for x in outputs[i]]).mean().item() / 2
                for i in [0, 1]
            ]
        )
        identity = sum(
            [
                torch.stack([x["identity"] for x in outputs[i]]).mean().item() / 2
                for i in [0, 1]
            ]
        )

        self.losses.append(avg_loss)
        self.G_mean_losses.append(G_mean_loss)
        self.D_mean_losses.append(D_mean_loss)
        self.validity.append(validity)
        self.reconstr.append(reconstr)
        self.identity.append(identity)
        return None
