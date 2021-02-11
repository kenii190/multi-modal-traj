import os
import sys
import argparse
import model
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl


def get_batch(motion_data, mb_size=128):
	rand_idx = np.random.randint(0, len(motion_data)-mb_size)
	return torch.from_numpy(motion_data[rand_idx : rand_idx+mb_size, :])

def main():
	#Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", default="../data/walkrun21143.pkl")
	args = parser.parse_args()

	#Parameter
	x_dim     = 63
	z_dim     = 8
	lr        = 1e-6
	mb_size   = 1024
	n_epoch   = 32
	beta      = 1e-8
	disp_step = 1000
	data_path = args.path
	save_dir  = "./save"
	device    = "cuda:0" if torch.cuda.is_available() else "cpu"

	#Data: normalized from [-1, 1] to [1, 0]
	motion_data = pkl.load(open(data_path, "rb"))
	motion_data = np.concatenate(motion_data, axis=0)
	motion_data = (motion_data + 1.0) / 2.0

	n_mb = len(motion_data) // mb_size

	#Model
	encoder = model.Encoder(x_dim, z_dim).to(device)
	decoder = model.Decoder(x_dim, z_dim).to(device)
	opt     = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr)
	print(encoder)
	print(decoder)

	#Load model
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	if os.path.exists(os.path.join(save_dir, "model.pt")):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(os.path.join(save_dir, "model.pt"), map_location=torch.device(device))
		encoder.load_state_dict(checkpoint["Encoder"])
		decoder.load_state_dict(checkpoint["Decoder"])
		start_epoch = checkpoint["epoch"]
		print("Done.")
	else:
		start_epoch = 0

	#Training
	for i_epoch in range(start_epoch, n_epoch):
		np.random.shuffle(motion_data)

		for i in range(n_mb):
			x = get_batch(motion_data).float().to(device)

			#Loss
			z_mean, z_logstd = encoder(x)
			z_sample = model.reparameterize(z_mean, z_logstd)
			x_recon  = decoder(z_sample)

			recon_loss = (x - x_recon).pow(2).mean()
			kl_loss    = (z_mean.pow(2) + z_logstd.exp().pow(2) - (z_logstd.exp().pow(2) + 1e-8).log()).sum(dim=1).mean()
			loss       = recon_loss + beta*kl_loss

			opt.zero_grad()
			loss.backward()
			opt.step()

			#Print
			if i % disp_step == 0:
				print("[{:3d} / {:3d}] [{:6d} / {:6d}] recon_loss = {:.6f}, kl_loss = {:.6f}, loss = {:.6f}, z_mean = {:.2f}, z_std = {:.2f}".format(
					i_epoch,
					n_epoch,
					i,
					n_mb,
					recon_loss.item(),
					kl_loss.item(),
					loss.item(),
					z_mean[0, 0].item(),
					z_logstd.exp()[0, 0].item()
				))

		#Save
		print("Saving the model ... ", end="")
		torch.save({
			"epoch": i_epoch,
			"Encoder": encoder.state_dict(),
			"Decoder": decoder.state_dict()
		}, os.path.join(save_dir, "model.pt"))
		print("Done.")
		print()


if __name__ == '__main__':
	main()