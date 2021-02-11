import sys
sys.path.insert(0, "..")

from model import PolicyNet, RNNEncoder
import torch
import os
import time
import circle_env
import numpy as np
import pickle as pkl


#-----------------------
# Sample a batch
#-----------------------
def sample_batch(s_traj, a_traj, mb_size=256):
	s_trajs  = []
	a_trajs  = []
	seq_lens = []
	max_len  = 0

	for i in range(mb_size):
		rand_idx = np.random.randint(0, len(s_traj))
		s_trajs.append(s_traj[rand_idx])
		a_trajs.append(a_traj[rand_idx])
		seq_lens.append(len(s_traj[rand_idx]))

		if max_len < len(s_traj[rand_idx]):
			max_len = len(s_traj[rand_idx])

	#mb_s_traj: (mb_size, max_len, s_dim)
	#mb_a_traj: (mb_size, max_len, a_dim)
	#mb_mask  : (mb_size, max_len)
	mb_s_traj = np.zeros((mb_size, max_len, len(s_traj[0][0])), dtype=np.float32) 
	mb_a_traj = np.zeros((mb_size, max_len, len(a_traj[0][0])), dtype=np.float32)
	mb_mask   = np.ones((mb_size, max_len), dtype=np.float32)

	for i in range(mb_size):
		mb_s_traj[i, :len(s_trajs[i]), :] = s_trajs[i]
		mb_a_traj[i, :len(a_trajs[i]), :] = a_trajs[i]
		mb_mask[i, seq_lens[i]:] = np.zeros((max_len - seq_lens[i]), dtype=np.float32)

	#mb_s_traj: (max_len, mb_size, s_dim)
	#mb_a_traj: (max_len, mb_size, a_dim)
	#mb_mask  : (max_len, mb_size)
	mb_s_traj = mb_s_traj.transpose((1, 0, 2))
	mb_a_traj = mb_a_traj.transpose((1, 0, 2))
	mb_mask   = mb_mask.transpose((1, 0))
		
	return mb_s_traj, mb_a_traj, np.concatenate([mb_s_traj, mb_a_traj], axis=2), mb_mask, seq_lens


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
	s_dim = circle_env.s_dim
	a_dim = circle_env.a_dim
	c_dim = 2

	#Load expert trajectories
	#----------------------------
	if os.path.exists(expert_path):
		s_traj, a_traj = pkl.load(open(expert_path, "rb"))
	else:
		print("ERROR: No expert trajectory file found")
		sys.exit(1)

	#Create model
	#----------------------------
	enc_net    = RNNEncoder(s_dim+a_dim, c_dim).to(device)
	policy_net = PolicyNet(s_dim, a_dim, c_dim, conti=True).to(device)
	opt        = torch.optim.Adam(list(enc_net.parameters()) + list(policy_net.parameters()), lr)
	print(enc_net)
	print(policy_net)

	#Load model
	#----------------------------
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	if os.path.exists(os.path.join(save_dir, "model.pt")):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(os.path.join(save_dir, "model.pt"), map_location=torch.device(device))
		policy_net.load_state_dict(checkpoint["PolicyNet"])
		enc_net.load_state_dict(checkpoint["EncoderNet"])
		start_it = checkpoint["it"]
		print("Done.")
	else:
		start_it = 0
		
	#Start training
	#----------------------------
	t_start = time.time()
	enc_net.train()
	policy_net.train()

	for it in range(start_it, start_it+n_iter):
		#Train
		mb_s_trajs, mb_a_trajs, mb_sa_trajs, mb_mask, seq_lens = sample_batch(s_traj, a_traj, mb_size)
		mb_s_trajs = torch.tensor(mb_s_trajs, dtype=torch.float32, device=device).contiguous()
		mb_a_trajs = torch.tensor(mb_a_trajs, dtype=torch.float32, device=device).contiguous()
		mb_sa_trajs = torch.tensor(mb_sa_trajs, dtype=torch.float32, device=device).contiguous()
		mb_mask = torch.tensor(mb_mask, dtype=torch.float32, device=device)

		c_mean, c_logstd, c_sample = enc_net(mb_sa_trajs, seq_lens)
		mb_a_logps, mb_ents = policy_net.evaluate(
			mb_s_trajs.view(-1, s_dim),
			mb_a_trajs.view(-1, a_dim),
			c_sample.unsqueeze(0).repeat(mb_s_trajs.size(0), 1, 1).view(-1, c_dim)
		)
		recon_loss = -(mb_a_logps.view(-1, mb_size) * mb_mask).sum(0).mean()
		kl_reg = (c_mean.pow(2) + c_logstd.exp().pow(2) - (c_logstd.exp().pow(2) + 1e-8).log()).sum(dim=1).mean()
		loss = recon_loss + 0.001*kl_reg

		opt.zero_grad()
		loss.backward()
		opt.step()

		#Print the result
		if it % disp_step == 0:
			print("[{:5d} / {:5d}] Elapsed time = {:.2f}, loss = {:.6f}, recon_loss = {:.6f}, kl_reg = {:.6f}".format(
				it, n_iter, time.time() - t_start, loss.item(), recon_loss.item(), kl_reg.item()
			))

		#Save model
		if it % save_step == 0:
			print("Saving the model ... ", end="")
			torch.save({
				"it": it,
				"PolicyNet": policy_net.state_dict(),
				"EncoderNet": enc_net.state_dict()
			}, os.path.join(save_dir, "model.pt"))
			print("Done.")
			print()

		#Save checkpoint
		if it % check_step == 0:
			print("Saving the checkpoint ... ", end="")
			torch.save({
				"it": it,
				"PolicyNet": policy_net.state_dict(),
				"EncoderNet": enc_net.state_dict()
			}, os.path.join(save_dir, "model{:d}.pt".format(it)))
			print("Done.")
			print()


if __name__ == '__main__':
	main()