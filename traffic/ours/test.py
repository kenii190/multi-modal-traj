import sys
sys.path.insert(0, "..")

from model import PolicyNet, EncoderNet
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
	#Parse arguments
	#----------------------------
	parser = argparse.ArgumentParser()
	parser.add_argument("--check", default=-1, type=int)
	parser.add_argument("--unlimit", default=False, action="store_true")
	args = parser.parse_args()

	#Parameters
	#----------------------------
	save_dir    = "./save"
	device      = "cuda:0" if torch.cuda.is_available() else "cpu"
	expert_path = "../expert_traj.pkl"

	#Load expert trajectories
	#----------------------------
	if os.path.exists(expert_path):
		s_real, a_real = pkl.load(open(expert_path, "rb"))
		sa_real = []

		for i in range(len(s_real)):
			sa_real.append(np.concatenate([s_real[i], a_real[i]], 1))
	else:
		print("ERROR: No expert trajectory file found")
		sys.exit(1)

	#Create environment
	#----------------------------
	env   = traffic_env.make(max_step=80)
	s_dim = traffic_env.s_dim
	a_dim = traffic_env.a_dim
	c_dim = 2
	traj_len = 4

	#Create model
	#----------------------------
	policy_net = PolicyNet(s_dim, a_dim, c_dim, conti=True).to(device)
	enc_net    = EncoderNet((s_dim+a_dim)*traj_len, c_dim).to(device)

	#Load model
	#----------------------------
	if args.check > 0:
		model_path = os.path.join(save_dir, "model{}.pt".format(args.check))
	else:
		model_path = os.path.join(save_dir, "model.pt")

	if os.path.exists(model_path):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(model_path, map_location=torch.device(device))
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
		cs[i, 0] = (0.0 * (len(cs) - 1.0 - i) + 0.0 * i) / (len(cs) - 1.0)
		cs[i, 1] = (-0.6 * (len(cs) - 1.0 - i) + 0.6 * i) / (len(cs) - 1.0)

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
	plt.title("Embedding")

	if not args.unlimit:
		plt.xlim(-2, 2)
		plt.ylim(-2, 2)

	cs = [[], [], []]
	colors = [
		[np.array([1.0, 0.0, 0.0])],
		[np.array([0.0, 1.0, 0.0])],
		[np.array([0.0, 0.0, 1.0])]
	]

	with torch.no_grad():
		for i in tqdm(range(len(cs))):
			for j in range(4):
				traj = sa_real[np.random.randint(i*1024, (i+1)*1024)]
				sa_traj = []

				for k in range(1, len(traj)//traj_len):
					sa_traj.append(traj[k*traj_len-traj_len : k*traj_len].flatten())

				c_sample = enc_net.get_code(
					torch.tensor(sa_traj, dtype=torch.float32, device=device)
				).cpu().numpy()

				cx = c_sample[:, 0]
				cy = c_sample[:, 1]
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