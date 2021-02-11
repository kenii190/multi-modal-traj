import sys
sys.path.append("..")

import gx
import os
import torch
import argparse
import pickle as pkl
import numpy as np
import renderer_manifold as renderer
from vae import model
from sklearn.decomposition import PCA


def main():
	#Parse arguments
	#----------------------------
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", default="../data/walkrun21143.pkl")
	args = parser.parse_args()

	#Parameters
	#--------------------------
	expert_path = args.path
	vae_path    = "../vae/save/model.pt"
	device      = "cuda:0" if torch.cuda.is_available() else "cpu"

	#Data
	#-----------------------
	motion_data = pkl.load(open(expert_path, "rb"))

	#normalized from [-1, 1] to [1, 0]
	for i in range(len(motion_data)):
		motion_data[i] = (motion_data[i] + 1.0) / 2.0

	#VAE Model
	#-----------------------
	x_dim = 63
	z_dim = 8

	encoder = model.Encoder(x_dim, z_dim).to(device)
	decoder = model.Decoder(x_dim, z_dim).to(device)
	print(encoder)
	print(decoder)

	if os.path.exists(vae_path):
		print("Loading VAE model ... ", end="")
		checkpoint = torch.load(vae_path, map_location=torch.device(device))
		encoder.load_state_dict(checkpoint["Encoder"])
		decoder.load_state_dict(checkpoint["Decoder"])
		print("Done.")
	else:
		print("Error: No VAE model saved")
		sys.exit(1)

	#PCA
	#--------------------------
	if os.path.exists("./pca_2d.pkl"):
		print("Loading PCA model ... ", end="")
		pca = pkl.load(open("./pca_2d.pkl", "rb"))
		print("Done.")
	else:
		print("Error: No PCA model saved")
		sys.exit(1)

	#Render loop
	#-----------------------
	renderer.player_window.max_data_idx = len(motion_data) - 1

	with torch.no_grad():
		while not renderer.window.should_close():
			renderer.window.start_frame()

			if renderer.player_window.load:
				x       = motion_data[renderer.player_window.data_idx]
				z_mean  = encoder.mean(torch.from_numpy(x).float().to(device))
				x_recon = decoder(z_mean).cpu().numpy()

				#denormalized from [0, 1] to [-pi, pi]
				x       = (x * 2.0 - 1.0) * np.pi
				x_recon = (x_recon * 2.0 - 1.0) * np.pi

				renderer.reload_player(
					x,
					x_recon, 
					pca.transform(z_mean.cpu().numpy())
				)

			if renderer.main_menu_bar.need_exit:
				break

			renderer.render_player(x, x_recon)
			renderer.render_screen()
			renderer.window.end_frame()
		
	renderer.terminate()


if __name__ == '__main__':
	main()