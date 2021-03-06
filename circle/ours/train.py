import sys
sys.path.insert(0, "..")

from multi_env import MultiEnv, make_env
from env_runner import EnvRunner
from model import PolicyNet, ValueNet, DiscriminatorNet, EncoderNet
from agent import PPO
import torch
import os
import time
import circle_env
import numpy as np
import pickle as pkl


#-----------------------
# Main function
#-----------------------
def main():
	#Parameters
	#----------------------------
	n_env          = 8
	n_step         = 128
	mb_size        = n_env*n_step
	sample_mb_size = 64
	sample_n_epoch = 4
	clip_val       = 0.2
	lamb           = 0.95
	gamma          = 0.99
	ent_weight     = 0.0
	max_grad_norm  = 0.5
	lr             = 1e-4
	n_iter         = 3000
	disp_step      = 30
	save_step      = 300
	check_step     = 1000
	save_dir       = "./save"
	device         = "cuda:0" if torch.cuda.is_available() else "cpu"
	expert_path    = "../expert_traj.pkl"

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

	#Create multiple environments
	#----------------------------
	env      = MultiEnv([make_env(i, rand_seed=int(time.time())) for i in range(n_env)])
	s_dim    = circle_env.s_dim
	a_dim    = circle_env.a_dim
	c_dim    = 2
	traj_len = 4

	runner = EnvRunner(
		env, 
		sa_real,
		s_dim, 
		a_dim,
		c_dim,
		traj_len,
		n_step, 
		gamma,
		lamb,
		device=device, 
		conti=True
	)

	#Create model
	#----------------------------
	policy_net = PolicyNet(s_dim, a_dim, c_dim, conti=True).to(device)
	value_net  = ValueNet(s_dim, c_dim).to(device)
	enc_net    = EncoderNet((s_dim+a_dim)*traj_len, c_dim).to(device)
	dis_net    = DiscriminatorNet(s_dim+a_dim, c_dim).to(device)
	agent      = PPO(
		policy_net, 
		value_net,
		enc_net,
		dis_net, 
		sa_real,
		a_dim, 
		traj_len,
		lr, 
		max_grad_norm, 
		ent_weight, 
		clip_val, 
		sample_n_epoch, 
		sample_mb_size, 
		mb_size,
		device=device
	)
	print(policy_net)
	print(value_net)
	print(enc_net)
	print(dis_net)

	#Load model
	#----------------------------
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	if os.path.exists(os.path.join(save_dir, "model.pt")):
		print("Loading the model ... ", end="")
		checkpoint = torch.load(os.path.join(save_dir, "model.pt"), map_location=torch.device(device))
		start_it = checkpoint["it"]
		policy_net.load_state_dict(checkpoint["PolicyNet"])
		value_net.load_state_dict(checkpoint["ValueNet"])
		enc_net.load_state_dict(checkpoint["EncoderNet"])
		dis_net.load_state_dict(checkpoint["DiscriminatorNet"])
		print("Done.")
	else:
		start_it = 0

	#Start training
	#----------------------------
	if torch.cuda.is_available():
		torch.cuda.synchronize()

	t_start = time.time()
	policy_net.train()
	value_net.train()
	enc_net.train()
	dis_net.train()

	for it in range(start_it, n_iter+1):
		#Run the environment
		with torch.no_grad():
			mb_obs, mb_actions, mb_cs, mb_old_a_logps, mb_values, mb_returns, mb_idxs, mb_sas = runner.run(
				policy_net, value_net, enc_net, dis_net
			)
			mb_advs = mb_returns - mb_values
			mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)

		#Train
		agent.train(
			policy_net, 
			value_net, 
			enc_net,
			dis_net,
			mb_obs, 
			mb_actions, 
			mb_cs,
			mb_values,
			mb_advs, 
			mb_returns,
			mb_old_a_logps,
			mb_idxs,
			mb_sas
		)

		#Print the result
		if it % disp_step == 0:
			if torch.cuda.is_available():
				torch.cuda.synchronize()
				
			n_sec = time.time() - t_start
			fps = int((it - start_it)*n_env*n_step / n_sec)
			mean_true_return, std_true_return, mean_return, std_return, mean_len = runner.get_performance()
			pg_loss, v_loss, ent, dis_loss, dis_real, dis_fake, kl_reg, siamese_loss = agent.get_eval()
			agent.lr_decay(it, n_iter)

			print("[{:5d} / {:5d}]".format(it, n_iter))
			print("----------------------------------")
			print("Timesteps        = {:d}".format((it - start_it) * mb_size))
			print("Elapsed time     = {:.2f} sec".format(n_sec))
			print("FPS              = {:d}".format(fps))
			print("actor loss       = {:.6f}".format(pg_loss))
			print("critic loss      = {:.6f}".format(v_loss))
			print("kl reg           = {:.6f}".format(kl_reg))
			print("siamese loss     = {:.6f}".format(siamese_loss))
			print("dis loss         = {:.6f}".format(dis_loss))
			print("entropy          = {:.6f}".format(ent))
			print("mean true return = {:.6f}".format(mean_true_return))
			print("mean return      = {:.6f}".format(mean_return))
			print("mean length      = {:.2f}".format(mean_len))
			print("dis_real         = {:.3f}".format(dis_real))
			print("dis_fake         = {:.3f}".format(dis_fake))
			print()

		#Save model
		if it % save_step == 0:
			print("Saving the model ... ", end="")
			torch.save({
				"it": it,
				"PolicyNet": policy_net.state_dict(),
				"ValueNet": value_net.state_dict(),
				"EncoderNet": enc_net.state_dict(),
				"DiscriminatorNet": dis_net.state_dict()
			}, os.path.join(save_dir, "model.pt"))
			print("Done.")
			print()

		#Save checkpoint
		if it % check_step == 0:
			print("Saving the checkpoint ... ", end="")
			torch.save({
				"it": it,
				"PolicyNet": policy_net.state_dict(),
				"ValueNet": value_net.state_dict(),
				"EncoderNet": enc_net.state_dict(),
				"DiscriminatorNet": dis_net.state_dict()
			}, os.path.join(save_dir, "model{:d}.pt".format(it)))
			print("Done.")
			print()

	env.close()


if __name__ == '__main__':
	main()