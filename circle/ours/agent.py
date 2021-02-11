from utils import linear_lr_decay
import torch
import torch.nn as nn
import numpy as np


#PPO Agent Class
class PPO:
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(
		self, 
		policy_net, 
		value_net, 
		enc_net,
		dis_net,
		sa_real,
		a_dim,
		traj_len=5,
		lr=1e-4, 
		max_grad_norm=0.5, 
		ent_weight=0.01,
		clip_val=0.2,
		sample_n_epoch=4,
		sample_mb_size=64,
		mb_size=1024,
		device="cuda:0"
	):
		self.opt_actor      = torch.optim.Adam(list(enc_net.parameters()) + list(policy_net.parameters()), lr)
		self.opt_critic     = torch.optim.Adam(value_net.parameters(), lr)
		self.opt_dis        = torch.optim.Adam(dis_net.parameters(), lr)
		self.sa_real        = sa_real
		self.a_dim          = a_dim
		self.traj_len       = traj_len
		self.lr             = lr
		self.max_grad_norm  = max_grad_norm
		self.ent_weight     = ent_weight
		self.clip_val       = clip_val
		self.sample_n_epoch = sample_n_epoch
		self.sample_mb_size = sample_mb_size
		self.sample_n_mb    = mb_size // sample_mb_size
		self.rand_idx       = np.arange(mb_size)
		self.criterion      = nn.BCELoss()
		self.ones_label     = torch.autograd.Variable(torch.ones((sample_mb_size, 1))).to(device)
		self.zeros_label    = torch.autograd.Variable(torch.zeros((sample_mb_size, 1))).to(device)
		self.device         = device

		self.mb_sa_real  = np.zeros((sample_mb_size, self.sa_real[0].shape[1]), dtype=np.float32)
		self.mb_sa_left  = np.zeros((sample_mb_size, self.sa_real[0].shape[1]*traj_len), dtype=np.float32)
		self.mb_sa_right = np.zeros((sample_mb_size, self.sa_real[0].shape[1]*traj_len), dtype=np.float32)
		self.mb_label    = np.zeros((sample_mb_size, 1), dtype=np.float32)

		for i in range(sample_mb_size//2):
			self.mb_label[i, 0] = 1

	#-----------------------
	# Train PPO
	#-----------------------
	def train(
		self, 
		policy_net, 
		value_net, 
		enc_net,
		dis_net,
		mb_obs, 
		mb_actions, 
		mb_cs,
		mb_old_values, 
		mb_advs, 
		mb_returns, 
		mb_old_a_logps, 
		mb_idxs,
		mb_sas
	):
		mb_obs         = torch.tensor(mb_obs, dtype=torch.float32, device=self.device)
		mb_actions     = torch.tensor(mb_actions, dtype=torch.float32, device=self.device)
		mb_cs          = torch.tensor(mb_cs, dtype=torch.float32, device=self.device)
		mb_old_values  = torch.tensor(mb_old_values, dtype=torch.float32, device=self.device)
		mb_advs        = torch.tensor(mb_advs, dtype=torch.float32, device=self.device)
		mb_returns     = torch.tensor(mb_returns, dtype=torch.float32, device=self.device)
		mb_old_a_logps = torch.tensor(mb_old_a_logps, dtype=torch.float32, device=self.device)
		mb_sas         = torch.tensor(mb_sas, dtype=torch.float32, device=self.device)

		#1. Train PPO
		for i in range(self.sample_n_epoch):
			np.random.shuffle(self.rand_idx)

			for j in range(self.sample_n_mb):
				sample_idx         = self.rand_idx[j*self.sample_mb_size : (j+1)*self.sample_mb_size]
				sample_obs         = mb_obs[sample_idx]
				sample_actions     = mb_actions[sample_idx]
				sample_old_values  = mb_old_values[sample_idx]
				sample_advs        = mb_advs[sample_idx]
				sample_returns     = mb_returns[sample_idx]
				sample_old_a_logps = mb_old_a_logps[sample_idx]
				sample_sas         = mb_sas[sample_idx]
				sample_cs          = mb_cs[sample_idx]

				c_mean, c_logstd, c_sample = enc_net(sample_sas)
				sample_a_logps, sample_ents = policy_net.evaluate(sample_obs, sample_actions, c_sample)
				sample_values = value_net(sample_obs, sample_cs)
				self.ent = sample_ents.mean()
				
				#Value loss
				v_pred_clip = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.clip_val, self.clip_val)
				v_loss1     = (sample_returns - sample_values).pow(2)
				v_loss2     = (sample_returns - v_pred_clip).pow(2)
				self.v_loss = torch.max(v_loss1, v_loss2).mean()

				#Policy gradient loss
				ratio    = (sample_a_logps - sample_old_a_logps).exp()
				pg_loss1 = -sample_advs * ratio
				pg_loss2 = -sample_advs * torch.clamp(ratio, 1.0-self.clip_val, 1.0+self.clip_val)

				#KL loss
				self.kl_reg = (c_mean.pow(2) + c_logstd.exp().pow(2) - (c_logstd.exp().pow(2) + 1e-8).log()).sum(dim=1).mean()
				
				#Siamese loss
				sample_sa_left, sample_sa_right, sample_label = self.get_siamese_batch()
				dist_square   = (enc_net.get_code(sample_sa_left) - enc_net.get_code(sample_sa_right)).pow(2).sum(dim=1, keepdim=True)
				similarity    = sample_label * dist_square
				dissimilarity = (1 - sample_label) * torch.clamp(0.5 - torch.sqrt(dist_square + 1e-8), min=0.0).pow(2)
				self.siamese_loss = (similarity + dissimilarity).mean()

				self.pg_loss = torch.max(pg_loss1, pg_loss2).mean() \
								- self.ent_weight*self.ent \
								+ 0.005*self.kl_reg \
								+ 0.5*self.siamese_loss

				#Train actor
				self.opt_actor.zero_grad()
				self.pg_loss.backward()
				nn.utils.clip_grad_norm_(policy_net.parameters(), self.max_grad_norm)
				self.opt_actor.step()

				#Train critic
				self.opt_critic.zero_grad()
				self.v_loss.backward()
				nn.utils.clip_grad_norm_(value_net.parameters(), self.max_grad_norm)
				self.opt_critic.step()

		#2. Train Discriminator
		np.random.shuffle(self.rand_idx)

		for i in range(self.sample_n_mb):
			sample_idx     = self.rand_idx[i*self.sample_mb_size : (i+1)*self.sample_mb_size]
			sample_obs     = mb_obs[sample_idx]
			sample_actions = mb_actions[sample_idx]
			sample_cs      = mb_cs[sample_idx]
			sample_idxs    = mb_idxs[sample_idx]

			mb_sa_real = self.get_sa_batch(sample_idxs)
			mb_sa_fake = torch.cat([sample_obs, sample_actions], 1)

			#Adversarial loss
			self.dis_real = dis_net(mb_sa_real, sample_cs)
			self.dis_fake = dis_net(mb_sa_fake, sample_cs)
			self.dis_loss = self.criterion(self.dis_real, self.ones_label) \
							+ self.criterion(self.dis_fake, self.zeros_label)

			self.opt_dis.zero_grad()
			self.dis_loss.backward()
			self.opt_dis.step()
		
	#-----------------------
	# Get evaluation
	#-----------------------
	def get_eval(self):
		return self.pg_loss.item(), \
				self.v_loss.item(), \
				self.ent.item(), \
				self.dis_loss.item(), \
				self.dis_real.mean().item(), \
				self.dis_fake.mean().item(), \
				self.kl_reg.item(), \
				self.siamese_loss.item()

	#-----------------------
	# Learning rate decay
	#-----------------------
	def lr_decay(self, it, n_it):
		linear_lr_decay(self.opt_actor, it, n_it, self.lr)
		linear_lr_decay(self.opt_critic, it, n_it, self.lr)

	#-----------------------
	# Get (s, a) batch
	#-----------------------
	def get_sa_batch(self, mb_idxs):
		for i in range(len(mb_idxs)):
			idx = mb_idxs[i]
			self.mb_sa_real[i, :] = self.sa_real[idx][np.random.randint(0, len(self.sa_real[idx]))]

		return torch.tensor(self.mb_sa_real, dtype=torch.float32, device=self.device)

	#-----------------------
	# Get siamese batch
	#-----------------------
	def get_siamese_batch(self):
		for i in range(self.sample_mb_size//2):
			rand_idx = np.random.randint(0, len(self.sa_real))
			n_data = len(self.sa_real[rand_idx])
			idx1 = np.random.randint(self.traj_len, n_data)
			idx2 = np.random.randint(self.traj_len, n_data)

			self.mb_sa_left[i]  = self.sa_real[rand_idx][idx1-self.traj_len:idx1].flatten()
			self.mb_sa_right[i] = self.sa_real[rand_idx][idx2-self.traj_len:idx2].flatten()

		for i in range(self.sample_mb_size//2, self.sample_mb_size):
			rand_idx_l, rand_idx_r = np.random.choice(len(self.sa_real), 2, replace=False)
			idx1 = np.random.randint(self.traj_len, len(self.sa_real[rand_idx_l]))
			idx2 = np.random.randint(self.traj_len, len(self.sa_real[rand_idx_r]))

			self.mb_sa_left[i]  = self.sa_real[rand_idx_l][idx1-self.traj_len:idx1].flatten()
			self.mb_sa_right[i] = self.sa_real[rand_idx_r][idx2-self.traj_len:idx2].flatten()

		return torch.tensor(self.mb_sa_left, dtype=torch.float32, device=self.device), \
				torch.tensor(self.mb_sa_right, dtype=torch.float32, device=self.device), \
				torch.tensor(self.mb_label, dtype=torch.float32, device=self.device)