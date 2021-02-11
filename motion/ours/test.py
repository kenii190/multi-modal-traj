from model import PolicyNet, EncoderNet
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import os
import sys
import argparse
import numpy as np
import pickle as pkl


#-----------------------
# Main function
#-----------------------
def main():
	#Parse arguments
	#----------------------------
	parser = argparse.ArgumentParser()
	parser.add_argument("--check", default=-1, type=int)
	parser.add_argument("--unlimit", default=False, action="store_true")
	parser.add_argument("--path", default="../data/sa_traj_valid_walkrun32.pkl")
	args = parser.parse_args()

	#Parameters
	#----------------------------
	save_dir    = "./save"
	device      = "cuda:0" if torch.cuda.is_available() else "cpu"
	expert_path = args.path

	#Load expert trajectories
	#----------------------------
	if os.path.exists(expert_path):
		s_real, a_real = pkl.load(open(expert_path, "rb"))
		s_dim   = s_real[0].shape[1]
		a_dim   = a_real[0].shape[1]
		c_dim   = 2
		sa_real = []

		for i in range(len(s_real)):
			sa_real.append(np.concatenate([s_real[i], a_real[i]], 1))
	else:
		print("ERROR: No expert trajectory file found")
		sys.exit(1)

	#Model
	#----------------------------
	policy_net = PolicyNet(s_dim, a_dim, c_dim, conti=True).to(device)
	enc_net    = EncoderNet(s_dim+a_dim, c_dim).to(device)
	print(policy_net)
	print(enc_net)

	if os.path.exists(os.path.join(save_dir, "model.pt")):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(os.path.join(save_dir, "model.pt"), map_location=torch.device(device))
		policy_net.load_state_dict(checkpoint["PolicyNet"])
		enc_net.load_state_dict(checkpoint["EncoderNet"])
		print("Done.")
	else:
		print("Error: No policy model saved")
		sys.exit(1)

	#Embedding
	#----------------------------
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.set_aspect(aspect=1.0)
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	plt.grid()

	if not args.unlimit:
		plt.xlim(-2, 2)
		plt.ylim(-2, 2)

	with torch.no_grad():
		for i in tqdm(range(16)):
			c_sample = enc_net.get_code(
				torch.tensor(sa_real[i], dtype=torch.float32, device=device)
			).cpu().numpy()

			cx = c_sample[:, 0]
			cy = c_sample[:, 1]
			ax.scatter(cx, cy, s=4, c=[np.array([1.0, 0.0, 0.0])])

		for i in tqdm(range(16, 32)):
			c_sample = enc_net.get_code(
				torch.tensor(sa_real[i], dtype=torch.float32, device=device)
			).cpu().numpy()

			cx = c_sample[:, 0]
			cy = c_sample[:, 1]
			ax.scatter(cx, cy, s=4, c=[np.array([0.0, 0.0, 1.0])])

	plt.show()


if __name__ == '__main__':
	main()