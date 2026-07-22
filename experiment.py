#%%
from data_sampler import sample_synthetic_extreme_data
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as dist
from models_torch import UExtVAE,SphereVAE



#%%
X, R, W = sample_synthetic_extreme_data(n=10_000)

print(X.shape)  # (10000, 5)
print(R.shape)  # (10000, 1)
print(W.shape)  # (10000, 5)

print(W.sum(dim=1).mean())  # approx 1
print(X.sum(dim=1, keepdim=True).sub(R).abs().max())  # approx 0


#%%
dataset = TensorDataset(X.float(), R.float(), W.float())

dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)

#%%
radius_vae = UExtVAE(trainable_prior=True)
optimizer_R = torch.optim.Adam(radius_vae.parameters(), lr=1e-3)

for epoch in range(1000):
    total_loss = 0.0

    for X_batch, R_batch, W_batch in dataloader:
        loss, logs = radius_vae.loss(R_batch)

        optimizer_R.zero_grad()
        loss.backward()
        optimizer_R.step()

        total_loss += loss.item()

    print(epoch, total_loss / len(dataloader), logs)

#%% 
sphere_vae = SphereVAE(
    input_dim=5,
    encoded_size=4,
    output_distribution="dirichlet"
)

optimizer_W = torch.optim.Adam(sphere_vae.parameters(), lr=1e-3)

for epoch in range(100):
    total_loss = 0.0

    for X_batch, R_batch, W_batch in dataloader:
        W_batch = torch.clamp(W_batch, min=1e-6)
        W_batch = W_batch / W_batch.sum(dim=1, keepdim=True)

        loss, logs = sphere_vae.loss(W_batch, R_batch)

        optimizer_W.zero_grad()
        loss.backward()
        optimizer_W.step()

        total_loss += loss.item()

    print(epoch, total_loss / len(dataloader), logs)

#%%
with torch.no_grad():
    n_gen = 1000

    # Rayon généré
    z_R = radius_vae.encoder.prior().sample((n_gen,))
    R_dist = radius_vae.decoder(z_R)
    R_gen = R_dist.sample()

    # Angle généré conditionnellement au rayon
    z_W = torch.randn(n_gen, sphere_vae.encoder.encoded_size)
    W_dist = sphere_vae.decoder(z_W, R_gen)
    W_gen = W_dist.sample()

    X_gen = R_gen * W_gen

#%%
# %%
