import sys
sys.path.append("..")

import os
import sys
import motion_env
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from vae import model
from tqdm import tqdm


def main():
	#Parameter
	x_dim     = 63
	z_dim     = 8
	data_path = sys.argv[1]
	save_dir  = "./vae/save"

	#Data: normalized from [-1, 1] to [1, 0]
	motion_data = pkl.load(open(data_path, "rb"))

	for i in range(len(motion_data)):
		motion_data[i] = (motion_data[i] + 1.0) / 2.0

	#Model
	encoder = model.Encoder(x_dim, z_dim).cuda()
	print(encoder)

	#Load model
	if os.path.exists(os.path.join(save_dir, "model.pt")):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(os.path.join(save_dir, "model.pt"))
		encoder.load_state_dict(checkpoint["Encoder"])
		print("Done.")
	else:
		raise RuntimeError("No model saved")

	#Generate (s, a) trajectories
	a_dim  = z_dim
	s_dim  = z_dim * 4
	state  = np.zeros((4, z_dim), dtype=np.float32)
	s_traj = []
	a_traj = []

	with torch.no_grad():
		for i in tqdm(range(len(motion_data))):
			states  = []
			actions = []
			z_traj  = encoder.mean(torch.FloatTensor(motion_data[i]).cuda()).cpu().numpy()

			if len(z_traj) < 5:
				continue

			for j in range(4):
				state[j, :] = z_traj[j]

			for j in range(4, len(z_traj)-1):
				states.append(state.flatten())
				actions.append((z_traj[j] - state[3]))

				for k in range(3):
					state[k, :] = state[k+1, :]

				state[3, :] = z_traj[j]

			s_traj.append(np.array(states, dtype=np.float32))
			a_traj.append(np.array(actions, dtype=np.float32))

	#Save data
	print("Saving the data ... ", end="")
	pkl.dump((s_traj, a_traj), open("./sa_traj.pkl", "wb"))
	print("Done.")


if __name__ == '__main__':
	main()