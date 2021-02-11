import sys
sys.path.insert(0, "..")

from model import PolicyNet
import torch
import os
import time
import traffic_env
import numpy as np
import pickle as pkl


#-----------------------
# Sample a batch
#-----------------------
def sample_batch(s_traj, a_traj, mb_size=256):
	rand_idx = np.arange(len(s_traj))
	np.random.shuffle(rand_idx)

	return s_traj[rand_idx[:mb_size]], a_traj[rand_idx[:mb_size]]

#-----------------------
# Main
#-----------------------
def main():
	#Parameters
	#----------------------------
	mb_size     = 256
	lr          = 1e-5
	n_iter      = 100000
	disp_step   = 500
	save_step   = 5000
	check_step  = 10000
	save_dir    = "./save"
	device      = "cuda:0" if torch.cuda.is_available() else "cpu"
	expert_path = "../expert_traj.pkl"

	#Create environment
	#----------------------------
	s_dim = traffic_env.s_dim
	a_dim = traffic_env.a_dim
	c_dim = 2

	#Load expert trajectories
	#----------------------------
	if os.path.exists(expert_path):
		s_traj, a_traj = pkl.load(open(expert_path, "rb"))
		s_traj = np.concatenate(s_traj, 0)
		a_traj = np.concatenate(a_traj, 0)
	else:
		print("ERROR: No expert trajectory file found")
		sys.exit(1)


	#Create model
	#----------------------------
	policy_net = PolicyNet(s_dim, a_dim, c_dim, conti=True).to(device)
	opt        = torch.optim.Adam(policy_net.parameters(), lr)

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	if os.path.exists(os.path.join(save_dir, "model.pt")):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(os.path.join(save_dir, "model.pt"), map_location=torch.device(device))
		policy_net.load_state_dict(checkpoint["PolicyNet"])
		start_it = checkpoint["it"]
		print("Done.")
	else:
		start_it = 0

	#Start training
	#----------------------------
	t_start = time.time()
	policy_net.train()

	for it in range(start_it, n_iter):
		#Train
		mb_obs, mb_actions  = sample_batch(s_traj, a_traj, mb_size)
		mb_a_logps, mb_ents = policy_net.evaluate(
			torch.tensor(mb_obs, dtype=torch.float32, device=device), 
			torch.tensor(mb_actions, dtype=torch.float32, device=device),
			torch.tensor(np.random.randn(mb_size, c_dim), dtype=torch.float32, device=device)
		)
		loss = -mb_a_logps.mean()

		opt.zero_grad()
		loss.backward()
		opt.step()

		#Print the result
		if it % disp_step == 0:
			print("[{:5d} / {:5d}] Elapsed time = {:.2f}, loss = {:.6f}".format(
				it, n_iter, time.time() - t_start, loss.item())
			)

		#Save model
		if it % save_step == 0:
			print("Saving the model ... ", end="")
			torch.save({
				"it": it,
				"PolicyNet": policy_net.state_dict()
			}, os.path.join(save_dir, "model.pt"))
			print("Done.")
			print()

		#Save checkpoint
		if it % check_step == 0:
			print("Saving the checkpoint ... ", end="")
			torch.save({
				"it": it,
				"PolicyNet": policy_net.state_dict()
			}, os.path.join(save_dir, "model{:d}.pt".format(it)))
			print("Done.")
			print()


if __name__ == '__main__':
	main()