import torch
import torch.nn as nn
import torch.optim as optim

# TODO(Shihan): The following parameters are copied from the AURORA paper. 
#   See https://arxiv.org/pdf/2504.01915 Table 7
#   We can move them into some config yaml later

input_dim = 1024       # Pi0 embedding size (after aggregating)
hidden_dim = 128       # Might want more since our data is higher-dimension
latent_dim = 10        # How many measures
seq_len = None         # Not sure. Depends on task.
batch_size = 128
num_epochs = 200
learning_rate = 0.01

# --- LSTM VAE Autoencoder ---
class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMVAE, self).__init__()

        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.outputs = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_n = h_n[-1]  # Take last layer's hidden state
        mu = self.hidden_to_mu(h_n)
        logvar = self.hidden_to_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        hidden = self.latent_to_hidden(z).unsqueeze(0)  # (1, batch, hidden)
        cell = torch.zeros_like(hidden)
        decoder_input = torch.zeros((z.size(0), seq_len, input_dim)).to(z.device)
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        output = self.outputs(decoder_output)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, x.size(1))
        return x_recon, mu, logvar

# --- Loss function ---
def vae_loss(x, x_recon, mu, logvar):
    recon_loss = nn.MSELoss()(x_recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_loss, recon_loss, kl_loss

# # --- Dummy data loader ---
# def generate_dummy_data(num_batches=100):
#     for _ in range(num_batches):
#         yield torch.randn(batch_size, seq_len, input_dim)

# # --- Training ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = LSTMVAE(input_dim, hidden_dim, latent_dim).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     total_loss = 0
#     for batch in generate_dummy_data():
#         batch = batch.to(device)
#         x_recon, mu, logvar = model(batch)
#         loss, recon, kl = vae_loss(batch, x_recon, mu, logvar)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f}")
