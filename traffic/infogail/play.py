import sys
sys.path.insert(0, "..")

from model import PolicyNet
import torch
import os
import traffic_env
import argparse
import numpy as np


#-----------------------
# Main function
#-----------------------
def main():
	#Parse arguments
	#----------------------------
	parser = argparse.ArgumentParser()
	parser.add_argument("--check", default=-1, type=int)
	parser.add_argument("--z1", default=0.0, type=float)
	parser.add_argument("--z2", default=0.0, type=float)
	parser.add_argument("--color", default="r")
	args = parser.parse_args()

	#Parameters
	#----------------------------
	save_dir = "./save"
	device   = "cuda:0" if torch.cuda.is_available() else "cpu"

	#Create environment
	#----------------------------
	env   = traffic_env.make(max_step=80)
	s_dim = traffic_env.s_dim
	a_dim = traffic_env.a_dim
	c_dim = 2

	if args.color == "r":
		env.color = [np.array([1.0, 0.0, 0.0])]
	elif args.color == "g":
		env.color = [np.array([0.0, 1.0, 0.0])]
	else:
		env.color = [np.array([0.0, 0.0, 1.0])]

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

	#Start playing
	#----------------------------
	policy_net.eval()
	c = np.array([[args.z1, args.z2]], dtype=np.float32)

	with torch.no_grad():
		for it in range(3):
			ob  = env.reset()
			ret = 0

			while True:
				env.render()
				action = policy_net.action_step(
					torch.tensor(np.expand_dims(ob, axis=0), dtype=torch.float32, device=device), 
					torch.tensor(c, dtype=torch.float32, device=device),
					deterministic=True
				)
				ob, reward, done, info = env.step(action.cpu().numpy()[0])
				ret += reward

				if done:
					print("return = {:.4f}".format(ret))
					break

	env.close()


if __name__ == '__main__':
	main()