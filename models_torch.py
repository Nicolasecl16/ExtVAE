import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


EPS = 1e-3


def gamma_kl(q_conc, q_rate, p_conc, p_rate):
    """
    KL(Gamma(q_conc, q_rate) || Gamma(p_conc, p_rate))
    PyTorch parametrization: concentration, rate.
    """
    q = dist.Gamma(q_conc, q_rate)
    p = dist.Gamma(p_conc, p_rate)
    return dist.kl_divergence(q, p)


class UEncoder(nn.Module):
    """
    Encoder for the radius VAE.

    q_phi(U | input) = Gamma(concentration, rate)
    prior p(U) = Gamma(prior_c, 1)

    prior_c is trainable.
    """

    def __init__(self, ndim=1, trainable_prior=True):
        super().__init__()

        self.ndim = ndim
        self.trainable_prior = trainable_prior
        self.alpha = 1.5

        self.dense1 = nn.Linear(1, 32)
        self.dense2 = nn.Linear(32, 32)
        self.dense3 = nn.Linear(32, 2)

        nn.init.uniform_(self.dense1.weight, 0.01, 0.02)
        nn.init.uniform_(self.dense2.weight, 0.01, 0.02)
        nn.init.uniform_(self.dense3.bias, 3.0, 5.5)

        if trainable_prior:
            init = torch.empty(ndim).uniform_(3.5, 5.5)
            self.raw_prior_c = nn.Parameter(torch.log(torch.exp(init) - 1.0))
        else:
            self.register_buffer("fixed_prior_c", torch.tensor([self.alpha]))

        self.last_q = None

    @property
    def prior_c(self):
        if self.trainable_prior:
            return F.softplus(self.raw_prior_c) + EPS
        return self.fixed_prior_c

    def prior(self):
        rate = torch.ones_like(self.prior_c)
        return dist.Gamma(self.prior_c, rate)

    def forward(self, inputs):
        x = F.relu(self.dense1(inputs))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))

        concentration = torch.abs(x[..., 0:1]) + EPS
        rate = torch.abs(x[..., 1:2]) + EPS

        q = dist.Gamma(concentration, rate)
        self.last_q = q
        return q

    def kl_loss(self):
        if self.last_q is None:
            raise RuntimeError("You must call forward before kl_loss().")

        p_conc = self.prior_c.view(1, -1)
        p_rate = torch.ones_like(p_conc)

        kl = gamma_kl(
            self.last_q.concentration,
            self.last_q.rate,
            p_conc,
            p_rate
        )

        return kl.mean()


class UDecoder(nn.Module):
    """
    Decoder for the radius VAE.

    p_theta(input | U) = Gamma(alpha(U), beta(U))
    """

    def __init__(self):
        super().__init__()

        self.dense1 = nn.Linear(1, 32)
        self.dense2 = nn.Linear(32, 32)
        self.dense31 = nn.Linear(32, 1)
        self.dense32 = nn.Linear(32, 1)

    def forward(self, u):
        u = u.view(-1, 1)
        y = 1.0 / (u + EPS)

        x = F.relu(self.dense1(y))
        x = F.relu(self.dense2(x))

        alpha = torch.abs(self.dense31(x) * u) + EPS
        beta = torch.abs(self.dense32(x) / (y ** 2 + EPS)) + EPS

        return dist.Gamma(alpha, beta)


class UExtVAE(nn.Module):
    """
    Radius VAE.
    """

    def __init__(self, trainable_prior=True):
        super().__init__()
        self.encoder = UEncoder(ndim=1, trainable_prior=trainable_prior)
        self.decoder = UDecoder()

    def forward(self, inputs, sample=True):
        q_u = self.encoder(inputs)

        if sample:
            u = q_u.rsample()
        else:
            u = q_u.mean

        p_x = self.decoder(u)
        return p_x

    def loss(self, inputs):
        p_x = self.forward(inputs, sample=True)

        reconstruction_loss = -p_x.log_prob(inputs).mean()
        kl = self.encoder.kl_loss()

        return reconstruction_loss + kl, {
            "reconstruction_loss": reconstruction_loss.detach(),
            "kl": kl.detach(),
            "prior_c": self.encoder.prior_c.detach()
        }
    

class SphereEncoder(nn.Module):
    """
    Encoder for the radius-conditioned angular VAE.

    q_phi(z | angle, radius) = Normal(mu, sigma)
    """

    def __init__(self, input_dim=5, encoded_size=4):
        super().__init__()

        self.input_dim = input_dim
        self.encoded_size = encoded_size

        self.dense1 = nn.Linear(input_dim + 1, 8)
        self.dense2 = nn.Linear(8, 8)
        self.dense3 = nn.Linear(8, 2 * encoded_size)

        self.last_mu = None
        self.last_std = None

    def forward(self, angle, radius):
        x = torch.cat([angle, radius / 10.0], dim=-1)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = torch.abs(x) + EPS

        params = self.dense3(x)

        mu = params[..., :self.encoded_size]
        raw_std = params[..., self.encoded_size:]
        std = F.softplus(raw_std) + EPS

        self.last_mu = mu
        self.last_std = std

        return dist.Independent(
            dist.Normal(mu, std),
            reinterpreted_batch_ndims=1
        )

    def kl_loss(self):
        if self.last_mu is None or self.last_std is None:
            raise RuntimeError("You must call forward before kl_loss().")

        q = dist.Independent(
            dist.Normal(self.last_mu, self.last_std),
            reinterpreted_batch_ndims=1
        )

        p = dist.Independent(
            dist.Normal(
                torch.zeros_like(self.last_mu),
                torch.ones_like(self.last_std)
            ),
            reinterpreted_batch_ndims=1
        )

        return dist.kl_divergence(q, p).mean()


class SphereDecoder(nn.Module):
    def __init__(
        self,
        encoded_size=4,
        output_dim=5,
        output_distribution="normal"
    ):
        super().__init__()

        assert output_distribution in ["normal", "dirichlet"]

        self.encoded_size = encoded_size
        self.output_dim = output_dim
        self.output_distribution = output_distribution

        self.dense1 = nn.Linear(encoded_size, 5)
        self.dense2 = nn.Linear(6, 10)

        if output_distribution == "normal":
            self.dense_out = nn.Linear(10, 2 * output_dim)
        else:
            self.dense_out = nn.Linear(10, output_dim)
            nn.init.uniform_(self.dense_out.bias, 2.0, 3.0)

    def forward(self, z, radius):
        radius_cond = 1.0 / (1.0 + radius)

        x = F.relu(self.dense1(z))
        x = torch.cat([x, radius_cond], dim=-1)
        x = F.relu(self.dense2(x))

        params = self.dense_out(x)

        if self.output_distribution == "normal":
            mu = params[..., :self.output_dim]
            raw_std = params[..., self.output_dim:]
            std = F.softplus(raw_std) + EPS

            return dist.Independent(
                dist.Normal(mu, std),
                reinterpreted_batch_ndims=1
            )

        if self.output_distribution == "dirichlet":
            concentration = F.softplus(params) + EPS

            return dist.Dirichlet(concentration)

class SphereVAE(nn.Module):
    def __init__(
        self,
        input_dim=5,
        encoded_size=4,
        output_distribution="normal"
    ):
        super().__init__()

        self.encoder = SphereEncoder(
            input_dim=input_dim,
            encoded_size=encoded_size
        )

        self.decoder = SphereDecoder(
            encoded_size=encoded_size,
            output_dim=input_dim,
            output_distribution=output_distribution
        )

    def forward(self, angle, radius, sample=True):
        q_z = self.encoder(angle, radius)
        z = q_z.rsample() if sample else q_z.mean
        return self.decoder(z, radius)

    def loss(self, angle, radius):
        p_angle = self.forward(angle, radius, sample=True)

        reconstruction_loss = -p_angle.log_prob(angle).mean()
        kl = self.encoder.kl_loss()

        return reconstruction_loss + kl, {
            "reconstruction_loss": reconstruction_loss.detach(),
            "kl": kl.detach()
        }
    
class ConvSphereEncoder(nn.Module):
    def __init__(self, encoded_size=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1),   # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), # 16 -> 8
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512 * 8 * 8, encoded_size)
        self.fc_logstd = nn.Linear(512 * 8 * 8, encoded_size)

        self.last_mu = None
        self.last_std = None

    def forward(self, angle, radius):
        """
        angle:  (B, 1, 256, 256)
        radius: (B, 1)
        """

        B, _, H, W = angle.shape

        radius_map = radius.view(B, 1, 1, 1).expand(B, 1, H, W)
        radius_map = torch.log1p(radius_map) / 10.0

        x = torch.cat([angle, radius_map], dim=1)

        h = self.conv(x)
        h = h.view(B, -1)

        mu = self.fc_mu(h)
        std = F.softplus(self.fc_logstd(h)) + EPS

        self.last_mu = mu
        self.last_std = std

        return dist.Independent(
            dist.Normal(mu, std),
            reinterpreted_batch_ndims=1
        )

    def kl_loss(self):
        q = dist.Independent(
            dist.Normal(self.last_mu, self.last_std),
            reinterpreted_batch_ndims=1
        )

        p = dist.Independent(
            dist.Normal(
                torch.zeros_like(self.last_mu),
                torch.ones_like(self.last_std)
            ),
            reinterpreted_batch_ndims=1
        )

        return dist.kl_divergence(q, p).mean()
    
class ConvSphereDecoder(nn.Module):
    def __init__(
        self,
        encoded_size=128,
        output_shape=(1, 256, 256),
        output_distribution="normal"
    ):
        super().__init__()

        assert output_distribution in ["normal", "dirichlet"]

        self.output_shape = output_shape
        self.output_distribution = output_distribution
        C, H, W = output_shape

        self.fc = nn.Linear(encoded_size + 1, 512 * 8 * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
        )

        if output_distribution == "normal":
            self.out = nn.ConvTranspose2d(32, 2 * C, 4, 2, 1)
        else:
            self.out = nn.ConvTranspose2d(32, C, 4, 2, 1)

    def forward(self, z, radius):
        r = torch.log1p(radius) / 10.0
        h = torch.cat([z, r], dim=-1)

        h = self.fc(h)
        h = h.view(z.shape[0], 512, 8, 8)

        h = self.deconv(h)
        params = self.out(h)

        if self.output_distribution == "normal":
            C = self.output_shape[0]

            mu = params[:, :C]
            raw_std = params[:, C:]
            std = F.softplus(raw_std) + EPS

            return dist.Independent(
                dist.Normal(mu, std),
                reinterpreted_batch_ndims=3
            )

        if self.output_distribution == "dirichlet":
            concentration = F.softplus(params) + EPS

            concentration = concentration.view(
                concentration.shape[0], -1
            )

            return dist.Dirichlet(concentration)
        
class ConvSphereVAE(nn.Module):
    def __init__(
        self,
        encoded_size=128,
        output_shape=(1, 256, 256),
        output_distribution="normal"
    ):
        super().__init__()

        self.encoder = ConvSphereEncoder(
            encoded_size=encoded_size
        )

        self.decoder = ConvSphereDecoder(
            encoded_size=encoded_size,
            output_shape=output_shape,
            output_distribution=output_distribution
        )

        self.output_shape = output_shape
        self.output_distribution = output_distribution

    def forward(self, angle, radius, sample=True):
        q_z = self.encoder(angle, radius)
        z = q_z.rsample() if sample else q_z.mean
        return self.decoder(z, radius)

    def loss(self, angle, radius):
        p_angle = self.forward(angle, radius, sample=True)

        if self.output_distribution == "normal":
            reconstruction_loss = -p_angle.log_prob(angle).mean()

        elif self.output_distribution == "dirichlet":
            target = angle.view(angle.shape[0], -1)
            target = target + EPS
            target = target / target.sum(dim=-1, keepdim=True)

            reconstruction_loss = -p_angle.log_prob(target).mean()

        kl = self.encoder.kl_loss()

        return reconstruction_loss + kl, {
            "reconstruction_loss": reconstruction_loss.detach(),
            "kl": kl.detach()
        }



EPS = 1e-3


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, beta=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings,
            1.0 / num_embeddings
        )

    def forward(self, z):
        """
        z: (B, D)
        """

        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            - 2 * z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )

        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)

        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)

        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, vq_loss, indices
    
class ConvSphereVQEncoder(nn.Module):
    def __init__(self, encoded_size=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1),    # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),   # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), # 16 -> 8
            nn.ReLU(),
        )

        self.fc = nn.Linear(512 * 8 * 8, encoded_size)

    def forward(self, angle, radius):
        """
        angle:  (B, 1, 256, 256)
        radius: (B, 1)
        """

        B, _, H, W = angle.shape

        r_cond = radius.view(B, 1, 1, 1)
        r_cond = 1.0 / (1.0 + r_cond)
        r_cond = r_cond.expand(B, 1, H, W)

        x = torch.cat([angle, r_cond], dim=1)

        h = self.conv(x)
        h = h.view(B, -1)

        z_e = self.fc(h)

        return z_e
    

class ConvSphereVQDecoder(nn.Module):
    def __init__(
        self,
        encoded_size=128,
        output_shape=(1, 256, 256),
        output_distribution="normal"
    ):
        super().__init__()

        assert output_distribution in ["normal", "dirichlet"]

        self.output_shape = output_shape
        self.output_distribution = output_distribution

        C, H, W = output_shape

        self.fc = nn.Linear(encoded_size + 1, 512 * 8 * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 64 -> 128
            nn.ReLU(),
        )

        if output_distribution == "normal":
            self.out = nn.ConvTranspose2d(32, 2 * C, 4, 2, 1)  # 128 -> 256
        else:
            self.out = nn.ConvTranspose2d(32, C, 4, 2, 1)

    def forward(self, z_q, radius):
        """
        z_q:    (B, encoded_size)
        radius: (B, 1)
        """

        r_cond = 1.0 / (1.0 + radius)

        h = torch.cat([z_q, r_cond], dim=-1)

        h = self.fc(h)
        h = h.view(z_q.shape[0], 512, 8, 8)

        h = self.deconv(h)
        params = self.out(h)

        if self.output_distribution == "normal":
            C = self.output_shape[0]

            mu = params[:, :C]
            raw_std = params[:, C:]
            std = F.softplus(raw_std) + EPS

            return dist.Independent(
                dist.Normal(mu, std),
                reinterpreted_batch_ndims=3
            )

        elif self.output_distribution == "dirichlet":
            concentration = F.softplus(params) + EPS
            concentration = concentration.view(concentration.shape[0], -1)

            return dist.Dirichlet(concentration)
        

class ConvSphereVQVAE(nn.Module):
    def __init__(
        self,
        encoded_size=128,
        output_shape=(1, 256, 256),
        output_distribution="normal",
        num_embeddings=512,
        beta=0.25
    ):
        super().__init__()

        self.encoder = ConvSphereVQEncoder(
            encoded_size=encoded_size
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=encoded_size,
            beta=beta
        )

        self.decoder = ConvSphereVQDecoder(
            encoded_size=encoded_size,
            output_shape=output_shape,
            output_distribution=output_distribution
        )

        self.output_shape = output_shape
        self.output_distribution = output_distribution

    def forward(self, angle, radius):
        z_e = self.encoder(angle, radius)

        z_q, vq_loss, indices = self.quantizer(z_e)

        p_angle = self.decoder(z_q, radius)

        return p_angle, vq_loss, indices

    def loss(self, angle, radius):
        p_angle, vq_loss, indices = self.forward(angle, radius)

        if self.output_distribution == "normal":
            reconstruction_loss = -p_angle.log_prob(angle).mean()

        elif self.output_distribution == "dirichlet":
            target = angle.view(angle.shape[0], -1)
            target = target + EPS
            target = target / target.sum(dim=-1, keepdim=True)

            reconstruction_loss = -p_angle.log_prob(target).mean()

        loss = reconstruction_loss + vq_loss

        return loss, {
            "reconstruction_loss": reconstruction_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "code_indices": indices.detach()
        }