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
		dis_net,
		enc_net,
		sa_real,
		a_dim,
		beta,
		lr=1e-4, 
		max_grad_norm=0.5, 
		ent_weight=0.01,
		gp_weight=1.0,
		clip_val=0.2,
		sample_n_epoch=4,
		sample_mb_size=64,
		mb_size=1024,
		device="cuda:0", 
		conti=False
	):
		self.opt_actor      = torch.optim.Adam(policy_net.parameters(), lr)
		self.opt_critic     = torch.optim.Adam(value_net.parameters(), lr)
		self.opt_dis        = torch.optim.Adam(dis_net.parameters(), lr)
		self.opt_enc        = torch.optim.Adam(enc_net.parameters(), lr)
		self.sa_real        = sa_real
		self.a_dim          = a_dim
		self.beta           = beta
		self.lr             = lr
		self.max_grad_norm  = max_grad_norm
		self.ent_weight     = ent_weight
		self.gp_weight      = gp_weight
		self.clip_val       = clip_val
		self.sample_n_epoch = sample_n_epoch
		self.sample_mb_size = sample_mb_size
		self.sample_n_mb    = mb_size // sample_mb_size
		self.rand_idx       = np.arange(mb_size)
		self.criterion      = nn.BCELoss()
		self.ones_label     = torch.autograd.Variable(torch.ones((sample_mb_size, 1))).to(device)
		self.zeros_label    = torch.autograd.Variable(torch.zeros((sample_mb_size, 1))).to(device)
		self.device         = device
		self.conti          = conti

	#-----------------------
	# Train PPO
	#-----------------------
	def train(
		self, 
		policy_net, 
		value_net, 
		dis_net,
		enc_net,
		mb_obs, 
		mb_actions,
		mb_cs, 
		mb_old_values, 
		mb_advs, 
		mb_returns, 
		mb_old_a_logps
	):
		mb_obs         = torch.from_numpy(mb_obs).to(self.device)
		mb_actions     = torch.from_numpy(mb_actions).to(self.device)
		mb_cs          = torch.from_numpy(mb_cs).to(self.device)
		mb_old_values  = torch.from_numpy(mb_old_values).to(self.device)
		mb_advs        = torch.from_numpy(mb_advs).to(self.device)
		mb_returns     = torch.from_numpy(mb_returns).to(self.device)
		mb_old_a_logps = torch.from_numpy(mb_old_a_logps).to(self.device)

		#1. Train PPO
		for i in range(self.sample_n_epoch):
			np.random.shuffle(self.rand_idx)

			for j in range(self.sample_n_mb):
				sample_idx         = self.rand_idx[j*self.sample_mb_size : (j+1)*self.sample_mb_size]
				sample_obs         = mb_obs[sample_idx]
				sample_actions     = mb_actions[sample_idx]
				sample_cs          = mb_cs[sample_idx]
				sample_old_values  = mb_old_values[sample_idx]
				sample_advs        = mb_advs[sample_idx]
				sample_returns     = mb_returns[sample_idx]
				sample_old_a_logps = mb_old_a_logps[sample_idx]

				sample_a_logps, sample_ents = policy_net.evaluate(sample_obs, sample_actions, sample_cs)
				sample_values = value_net(sample_obs)
				self.ent = sample_ents.mean()
				
				#PPO loss
				v_pred_clip = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.clip_val, self.clip_val)
				v_loss1     = (sample_returns - sample_values).pow(2)
				v_loss2     = (sample_returns - v_pred_clip).pow(2)
				self.v_loss = torch.max(v_loss1, v_loss2).mean()

				ratio    = (sample_a_logps - sample_old_a_logps).exp()
				pg_loss1 = -sample_advs * ratio
				pg_loss2 = -sample_advs * torch.clamp(ratio, 1.0-self.clip_val, 1.0+self.clip_val)
				self.pg_loss = torch.max(pg_loss1, pg_loss2).mean() - self.ent_weight*self.ent

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

			#Continuous: concat (s, a)
			if self.conti:
				mb_sa_fake = torch.cat([sample_obs, sample_actions], 1)
			
			#Discrete: concat (s, a_onehot)
			else:
				sample_actions_onehot = np.zeros([self.sample_mb_size, self.a_dim], dtype=np.float32)

				for j in range(self.sample_mb_size):
					sample_actions_onehot[j, sample_actions[j]] = 1

				mb_sa_fake = torch.cat([sample_obs, torch.tensor(sample_actions_onehot, dtype=torch.float32, device=self.device)], 1)

			mb_sa_real = self.sa_real[np.random.randint(0, len(self.sa_real), self.sample_mb_size), :]
			mb_sa_real = torch.tensor(mb_sa_real, dtype=torch.float32, device=self.device)
			mb_sa_real.requires_grad_()

			#Adversarial loss
			self.dis_real, z_mean_real, z_logstd_real = dis_net(mb_sa_real)
			self.dis_fake, z_mean_fake, z_logstd_fake = dis_net(mb_sa_fake)

			kl_real = self.kl_loss(z_mean_real, z_logstd_real).mean()
			kl_fake = self.kl_loss(z_mean_fake, z_logstd_fake).mean()
			self.avg_kl = 0.5 * (kl_real + kl_fake)

			self.dis_loss = self.criterion(self.dis_real, self.ones_label) \
							+ self.criterion(self.dis_fake, self.zeros_label) \
							+ self.beta*self.avg_kl

			self.opt_dis.zero_grad()
			self.dis_loss.backward()
			self.opt_dis.step()
			self.update_beta(self.avg_kl)

		#3. Train Encoder
		np.random.shuffle(self.rand_idx)

		for i in range(self.sample_n_mb):
			sample_idx     = self.rand_idx[i*self.sample_mb_size : (i+1)*self.sample_mb_size]
			sample_obs     = mb_obs[sample_idx]
			sample_actions = mb_actions[sample_idx]
			sample_cs      = mb_cs[sample_idx]

			#Continuous: concat (s, a)
			if self.conti:
				sample_sa = torch.cat([sample_obs, sample_actions], 1)
			
			#Discrete: concat (s, a_onehot)
			else:
				sample_actions_onehot = np.zeros([self.sample_mb_size, self.a_dim], dtype=np.float32)

				for j in range(self.sample_mb_size):
					sample_actions_onehot[j, sample_actions[j]] = 1

				sample_sa = torch.cat([sample_obs, torch.tensor(sample_actions_onehot, dtype=torch.float32, device=self.device)], 1)

			self.enc_loss = -enc_net.get_logp(sample_sa, sample_cs).mean()

			self.opt_enc.zero_grad()
			self.enc_loss.backward()
			self.opt_enc.step()

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
				self.avg_kl.item(), \
				self.enc_loss.item()

	#-----------------------
	# Learning rate decay
	#-----------------------
	def lr_decay(self, it, n_it):
		linear_lr_decay(self.opt_actor, it, n_it, self.lr)
		linear_lr_decay(self.opt_critic, it, n_it, self.lr)
		linear_lr_decay(self.opt_enc, it, n_it, self.lr)

	#-----------------------
	# Compute KL loss
	#-----------------------
	def kl_loss(self, mean, logstd):
		std = torch.exp(logstd)
		return torch.sum(-logstd + 0.5*(std**2 + mean**2), dim=-1) - 0.5*mean.shape[1]

	#-----------------------
	# Compute gradient square
	#-----------------------
	def compute_grad2(self, d_out, x_in):
		grad_d_out = torch.autograd.grad(
			outputs=d_out.sum(),
			inputs=x_in,
			create_graph=True,
			retain_graph=True,
			only_inputs=True
		)[0]
		grad_d_out2 = grad_d_out.pow(2)

		return grad_d_out2.view(x_in.size(0), -1).sum(1)

	#-----------------------
	# Update beta
	#-----------------------
	def update_beta(self, avg_kl, target_kl=0.1, beta_step=1e-5):
		with torch.no_grad():
			self.beta = self.beta - beta_step * (target_kl - avg_kl)