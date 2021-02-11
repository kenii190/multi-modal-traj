import sys
sys.path.insert(0, "..")

from model import PolicyNet
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import os
import circle_env
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
	save_dir = "./save"
	device   = "cuda:0" if torch.cuda.is_available() else "cpu"

	#Create environment
	#----------------------------
	env   = circle_env.make(max_step=128)
	s_dim = circle_env.s_dim
	a_dim = circle_env.a_dim 
	c_dim = 2

	#Create model
	#----------------------------
	policy_net = PolicyNet(s_dim, a_dim, c_dim, conti=True).to(device)

	if args.check > 0:
		model_path = os.path.join(save_dir, "model{}.pt".format(args.check))
	else:
		model_path = os.path.join(save_dir, "model.pt")

	if os.path.exists(model_path):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(model_path, map_location=torch.device(device))
		policy_net.load_state_dict(checkpoint["PolicyNet"])
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
	plt.xlim(-1, 1)
	plt.ylim(-1, 1)
	plt.title("Interpolation")

	cs = np.zeros((15, c_dim), dtype=np.float32)

	for i in range(len(cs)):
		cs[i, 0] = (-1.2 * (len(cs) - 1.0 - i) + 1.2 * i) / (len(cs) - 1.0)
		cs[i, 1] = (-1.2 * (len(cs) - 1.0 - i) + 1.2 * i) / (len(cs) - 1.0)

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

	#Reconstruction
	#----------------------------
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.set_aspect(aspect=1.0)
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	plt.grid()
	plt.xlim(-1, 1)
	plt.ylim(-1, 1)
	plt.title("Reconstruction")
	cs = np.array(cs)

	cs = np.zeros((3, c_dim), dtype=np.float32)

	cs[0] = np.array([-1, -1], dtype=np.float32)
	cs[1] = np.array([0, 0], dtype=np.float32)
	cs[2] = np.array([1, 1], dtype=np.float32)

	colors = [
		[np.array([1.0, 0.0, 0.0])], #red
		[np.array([0.0, 1.0, 0.0])], #green
		[np.array([0.0, 0.0, 1.0])]  #blue
	]

	for i in tqdm(range(len(colors))):
		c  = torch.tensor(cs[i : i+1], dtype=torch.float32, device=device)
		xs = []
		ys = []

		with torch.no_grad():
			for j in range(8):
				ob = env.reset()

				while True:
					action = policy_net.action_step(
						torch.tensor(np.expand_dims(ob, axis=0), dtype=torch.float32, device=device), 
						c,
						deterministic=True
					)
					ob, reward, done, info = env.step(action.cpu().detach().numpy()[0])

					if done:
						x, y = env.get_points()
						xs += x
						ys += y
						break

		ax.scatter(xs, ys, s=4, c=colors[i])

	env.close()
	plt.show()


if __name__ == '__main__':
	main()