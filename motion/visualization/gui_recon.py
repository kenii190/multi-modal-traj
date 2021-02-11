import sys
sys.path.append("..")

import gx
import os
import torch
import motion_env
import argparse
import pickle as pkl
import numpy as np
import renderer_recon as renderer
import vae.model as vae_model
from sklearn.decomposition import PCA


def main():
	#Parse arguments
	#----------------------------
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", default="ours")
	parser.add_argument("--check", default=-1, type=int)
	parser.add_argument("--path", default="../data/sa_traj_walkrun21143.pkl")
	parser.add_argument("--motion_path", default="../data/walkrun21143.pkl")
	args = parser.parse_args()

	#Parameters
	#--------------------------
	expert_path          = args.path
	motion_path          = args.motion_path
	vae_path             = "../vae/save/model.pt"
	motion_data_all_path = "../data/sa_traj_walkrun21143.pkl"
	device               = "cuda:0" if torch.cuda.is_available() else "cpu"

	if args.model == "ours":
		import ours.model as policy_model
		policy_path = "../ours/save/model.pt"
	elif args.model == "infogail":
		import infogail.model as policy_model
		policy_path = "../infogail/save/model.pt"
	elif args.model == "vae_bc":
		import vae_bc.model as policy_model
		policy_path = "../vae_bc/save/model.pt"
	else:
		print("ERROR: No such model")
		sys.exit(1)

	#Data
	#--------------------------
	motion_data = pkl.load(open(motion_path, "rb"))

	#normalized from [-1, 1] to [1, 0]
	for i in range(len(motion_data)):
		motion_data[i] = (motion_data[i] + 1.0) / 2.0

	#normalized from [-1, 1] to [1, 0]
	s_traj, a_traj = pkl.load(open(motion_data_all_path, "rb"))
	sa_real = []

	for i in range(len(s_traj)):
		sa_real.append(np.concatenate([s_traj[i], a_traj[i]], 1))

	s_traj = np.concatenate(s_traj, 0)

	if os.path.exists(expert_path):
		s, a = pkl.load(open(expert_path, "rb"))
		s_dim  = s[0].shape[1]
		a_dim  = a[0].shape[1]
		z_dim  = a_dim
		c_dim  = 2
	else:
		print("ERROR: No expert trajectory file found")
		sys.exit(1)

	#VAE Model
	#--------------------------
	x_dim   = 63
	encoder = vae_model.Encoder(x_dim, z_dim).to(device)
	decoder = vae_model.Decoder(x_dim, z_dim).to(device)
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

	#Policy Model
	#--------------------------
	policy_net = policy_model.PolicyNet(s_dim, a_dim, c_dim, conti=True).to(device)

	if args.model == "ours":
		enc_net = policy_model.EncoderNet(s_dim+a_dim, c_dim).to(device)
	elif args.model == "infogail":
		enc_net = policy_model.EncoderNet(s_dim+a_dim, c_dim, conti=True).to(device)
	elif args.model == "vae_bc":
		enc_net = policy_model.RNNEncoder(s_dim+a_dim, c_dim).to(device)

	print(policy_net)
	print(enc_net)

	if os.path.exists(policy_path):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(policy_path, map_location=torch.device(device))
		policy_net.load_state_dict(checkpoint["PolicyNet"])
		enc_net.load_state_dict(checkpoint["EncoderNet"])
		print("Done.")
	else:
		print("Error: No policy model saved")
		sys.exit(1)

	#PCA
	#--------------------------
	if os.path.exists("./pca_2d.pkl"):
		print("Loading PCA model ... ", end="")
		pca = pkl.load(open("./pca_2d.pkl", "rb"))
		print("Done.")
	else:
		with torch.no_grad():
			z_data = s_traj[:, :z_dim]
			pca = PCA(n_components=2)
			pca.fit(z_data)
			print("Saving PCA model ... ", end="")
			pkl.dump(pca, open("./pca_2d.pkl", "wb"))
			print("Done.")

	#Environment
	#--------------------------
	env    = motion_env.make(expert_path, max_step=4096)
	state  = env.reset()
	n_step = 0

	z_raw    = np.array([state[:z_dim]])
	z_pca    = pca.transform(z_raw)
	z_points = [gx.vec3(z_pca[0, 0], z_pca[0, 1], 0.0)]

	#denormalized from [0, 1] to [-pi, pi]
	with torch.no_grad():
		decoded_x = decoder(torch.FloatTensor(z_raw).to(device)).cpu().numpy()[0]
		decoded_x = (decoded_x * 2.0 - 1.0) * np.pi

		#denormalized from [0, 1] to [-pi, pi]
		x = motion_data[0]
		x = (x * 2.0 - 1.0) * np.pi

		#Encode traj
		renderer.player_window.cs[0, :] = enc_net.get_code(
			torch.tensor(sa_real[0], dtype=torch.float32, device=device)
		).cpu().numpy().mean(axis=0)

	#Render loop
	#--------------------------
	renderer.player_window.max_data_idx = len(motion_data) - 1

	with torch.no_grad():
		while not renderer.window.should_close():
			renderer.window.start_frame()

			if renderer.main_menu_bar.need_exit:
				break

			if renderer.render_player(x[n_step], decoded_x, z_points):
				action = policy_net.action_step(
					torch.FloatTensor(np.expand_dims(state, 0)).to(device), 
					torch.FloatTensor(renderer.player_window.cs).to(device),
					deterministic=True
				)
				state, reward, done, info = env.step(action.cpu().numpy()[0])
				n_step += 1

				if n_step >= len(x):
					n_step = 0

				if done or renderer.player_window.need_reset:
					state = env.reset()
					z_points.clear()
					renderer.player_window.frame_num = 0
					renderer.player_window.need_reset = False
					n_step = 0

					#denormalized from [0, 1] to [-pi, pi]
					x = motion_data[renderer.player_window.data_idx]
					x = (x * 2.0 - 1.0) * np.pi

					#Encode traj
					renderer.player_window.cs[0, :] = enc_net.get_code(
						torch.tensor(sa_real[renderer.player_window.data_idx], dtype=torch.float32, device=device)
					).cpu().numpy().mean(axis=0)

				z_raw = np.array([state[:z_dim]])
				z_pca = pca.transform(z_raw)
				z_points.append(gx.vec3(z_pca[0, 0], z_pca[0, 1], 0.0))

				if len(z_points) > 256:
					z_points.pop(0)

				#denormalized from [0, 1] to [-pi, pi]
				decoded_x = decoder(torch.FloatTensor(z_raw).to(device)).cpu().numpy()[0]
				decoded_x = (decoded_x * 2.0 - 1.0) * np.pi

			renderer.render_screen()
			renderer.window.end_frame()
		
	renderer.terminate()
	env.close()


if __name__ == '__main__':
	main()