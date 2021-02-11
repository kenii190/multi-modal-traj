import sys
sys.path.insert(0, "..")

from model import PolicyNet, RNNEncoder
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import os
import traffic_env
import argparse
import numpy as np
import pickle as pkl


#-----------------------
# Main function
#-----------------------
def main():
	#Parameters
	#----------------------------
	save_dir    = "./save"
	device      = "cuda:0" if torch.cuda.is_available() else "cpu"
	expert_path = "../expert_traj.pkl"

	#Create environment
	#----------------------------
	env   = traffic_env.make(max_step=80)
	s_dim = traffic_env.s_dim
	a_dim = traffic_env.a_dim 
	c_dim = 2

	#Load expert trajectories
	#----------------------------
	if os.path.exists(expert_path):
		s_traj, a_traj = pkl.load(open(expert_path, "rb"))
		s_dim   = s_traj[0].shape[1]
		a_dim   = a_traj[0].shape[1]
		c_dim   = 2
		sa_traj = []

		for i in range(len(s_traj)):
			sa_traj.append(np.expand_dims(np.concatenate([s_traj[i], a_traj[i]], axis=1), axis=0).transpose((1, 0, 2)))
	else:
		print("ERROR: No expert trajectory file found")
		sys.exit(1)

	#Create model
	#----------------------------
	enc_net    = RNNEncoder(s_dim+a_dim, c_dim).to(device)
	policy_net = PolicyNet(s_dim, a_dim, c_dim, conti=True).to(device)

	#Load model
	#----------------------------
	if os.path.exists(os.path.join(save_dir, "model.pt")):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(os.path.join(save_dir, "model.pt"), map_location=torch.device(device))
		policy_net.load_state_dict(checkpoint["PolicyNet"])
		enc_net.load_state_dict(checkpoint["EncoderNet"])
		print("Done.")
	else:
		print("Error: No model saved")

	#Interpolation
	#----------------------------
	policy_net.eval()
	plt.close()
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.set_aspect(aspect=1.0)
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	plt.grid()
	plt.xlim(0, 1)
	plt.ylim(-0.5, 0.5)
	plt.title("Interpolation")
	cs = np.zeros((15, c_dim), dtype=np.float32)

	for i in range(len(cs)):
		cs[i, 0] = (-1.0 * (len(cs) - 1.0 - i) + 1.0 * i) / (len(cs) - 1.0)
		cs[i, 1] = (-1.0 * (len(cs) - 1.0 - i) + 1.0 * i) / (len(cs) - 1.0)

	colors = [
		[np.array([1.0, 0.0, 0.0])], #red
		[np.array([1.0, 0.3, 0.1])],
		[np.array([1.0, 0.6, 0.2])], #orange
		[np.array([0.9, 0.75, 0.1])],
		[np.array([0.8, 0.8, 0.0])], #yellow
		[np.array([0.4, 0.9, 0.0])],
		[np.array([0.0, 1.0, 0.0])], #green
		[np.array([0.0, 0.9, 0.4])],
		[np.array([0.0, 0.8, 0.8])], #cyan
		[np.array([0.0, 0.4, 0.4])],
		[np.array([0.0, 0.0, 1.0])], #blue
		[np.array([0.4, 0.0, 0.9])],
		[np.array([0.8, 0.0, 0.8])], #purple
		[np.array([0.9, 0.0, 0.8])],
		[np.array([1.0, 0.3, 0.8])]  #pink
	]

	for i in tqdm(range(len(colors))):
		c  = torch.FloatTensor(cs[i : i+1]).to(device)
		xs = []
		ys = []

		for j in range(3):
			ob = env.reset()

			with torch.no_grad():
				while True:
					action = policy_net.action_step(
						torch.tensor(np.expand_dims(ob, axis=0), dtype=torch.float32, device=device), 
						c,
						deterministic=True
					)
					ob, reward, done, info = env.step(action.cpu().numpy()[0])

					if done:
						x, y = env.get_points()
						xs += x
						ys += y
						break

		ax.scatter(xs, ys, s=4, c=colors[i], label="z = ({:.1f}, {:.1f})".format(cs[i, 0], cs[i, 1]))

	#Embedding
	#----------------------------
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.set_aspect(aspect=1.0)
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	plt.grid()
	plt.xlim(-2, 2)
	plt.ylim(-2, 2)
	plt.title("Embedding")

	cs = [[], [], []]
	colors = [
		[np.array([1.0, 0.0, 0.0])],
		[np.array([0.0, 1.0, 0.0])],
		[np.array([0.0, 0.0, 1.0])]
	]

	with torch.no_grad():
		for i in tqdm(range(len(cs))):
			for j in range(64):
				rand_idx = np.random.randint(i*1024, (i+1)*1024)

				c_sample = enc_net.get_code(
					torch.tensor(sa_traj[rand_idx], dtype=torch.float32, device=device),
					deterministic=False
				).cpu().numpy()

				cx = c_sample[:, 0]
				cy = c_sample[:, 1]

				if j < 3:
					cs[i].append(c_sample[np.random.randint(0, len(c_sample))])

				ax.scatter(cx, cy, s=4, c=colors[i])

	#Reconstruction
	#----------------------------
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.set_aspect(aspect=1.0)
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	plt.grid()
	plt.xlim(0, 1)
	plt.ylim(-0.5, 0.5)
	plt.title("Reconstruction")
	cs = np.array(cs)

	for i in tqdm(range(len(cs))):
		for j in range(len(cs[i])):
			c  = torch.tensor(cs[i][j : j+1], dtype=torch.float32, device=device)
			xs = []
			ys = []
			ob = env.reset()

			with torch.no_grad():
				while True:
					action = policy_net.action_step(
						torch.tensor(np.expand_dims(ob, axis=0), dtype=torch.float32, device=device), 
						c,
						deterministic=True
					)
					ob, reward, done, info = env.step(action.cpu().numpy()[0])

					if done:
						x, y = env.get_points()
						xs += x
						ys += y
						break

			ax.scatter(xs, ys, s=4, c=colors[i], label="z = ({:.1f}, {:.1f})".format(cs[i][j, 0], cs[i][j, 1]))

	env.close()
	plt.show()


if __name__ == '__main__':
	main()