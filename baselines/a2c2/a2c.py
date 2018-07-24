import os.path as osp
import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.runners import AbstractEnvRunner
from baselines.common import tf_util

from baselines.a2c2.utils import discount_with_dones
from baselines.a2c2.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c2.utils import cat_entropy, mse
from baselines.a2c2.utils import soc_loss

from baselines.a2c2.policies import LstmPolicy

class Model(object):

    # TODO
    # DEBUG
    # Experiment with different learning rates
    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstates=None,
            ent_coef=0.01, vf_coef=0.5, sc_coef=0.1, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.make_session() #add log_devices=True argument for logging
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])
        if nstates is not None:
            S = tf.placeholder(tf.float32, [nbatch, nstates])
        if policy == LstmPolicy:
            # S = S[:, :nstates//2] # take only c; exlcude h from soc penalty
            # nstates = nstates//2
            ctanh = tf.tanh(S)
            cavg = ctanh.reshape([nenv, nsteps, states])
            cavg = tf.reduce_mean(cavg, axis=1)

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        if sc_coef is not None:
            sc_loss = tf.reduce_mean(soc_loss(cavg))
        
        loss = pg_loss - entropy*ent_coef + vf_loss*vf_coef
        if sc_coef is not None:
            loss += sc_loss*sc_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        
        # tensorboard logging
        summ = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('/tmp/openai/tensorboard')
        summary_writer.add_graph(sess.graph)
        
        # TODO
        # DEBUG
        # Added optional arguments log, i to train

        def train(obs, states, rewards, masks, actions, values, log=False, i=None):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if nstates is not None:
                td_map[S] = states
            if states is not None:
                # convert to expected input format
                logged_states = states.reshape([nenv, nsteps, nstates])
                input_states = logged_states[:, 0, :]
                td_map[train_model.S] = input_states
                td_map[train_model.M] = masks
            # include soc_loss in fetches if appropriate
            if sc_coef is not None:    
                policy_loss, value_loss, policy_entropy, soc_loss, _, s = sess.run(
                    [pg_loss, vf_loss, entropy, sc_loss, _train, summ],
                    td_map
                )
            # TODO
            # DEBUG
            # Added grads for debugging
            # Add summ for debugging
            else:
                policy_loss, value_loss, policy_entropy, _, s = sess.run(
                    [pg_loss, vf_loss, entropy, _train, summ],
                    td_map
                )
                soc_loss = None
            if log:
                assert i is not None
                summary_writer.add_summary(s, i)
            return policy_loss, value_loss, policy_entropy, soc_loss
            
        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma

    def run(self):
        mb_obs, mb_states, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[],[]
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
            mb_states.append(states)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        if mb_states[0] is not None: # reshape if states exist
            mb_states = np.asarray(mb_states, dtype=np.float32).swapaxes(1, 0)
            shape = shape(mb_states)
            mb_states = mb_states.reshape([shape[0]*shape[1], shape[2]])
        else: # if states are None's, return single None instead of list of None's
            mb_states = None
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

# TODO
# DEBUGGING: REVERT AFTER DEBUGGING
# changed lr to higher:
### LR FROM: 7e-4
### LR TO: 5e-3
def learn(policy, env, seed, nsteps=5, nstates=512, 
# changed to 200 during debugging
# total_timesteps=int(80e6),
total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, sc_coef=0.1, max_grad_norm=0.5, lr=5e-3, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99,
# set log_interval to 2 for debugging
# set log_interval to 100 for run
log_interval=100):
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstates=nstates, ent_coef=ent_coef, vf_coef=vf_coef, sc_coef=sc_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    # simple logging
    timesteps = []
    evs = []
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run()
        if update % log_interval == 0 or update == 1:
            policy_loss, value_loss, policy_entropy, sc_loss = model.train(obs, states, rewards, masks, actions, values, log=True, i=update)
        else:
            policy_loss, value_loss, policy_entropy, sc_loss = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            if sc_coef is not None and sc_coef != 0.0:
                logger.record_tabular("soc_loss", float(sc_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
            # light logging
            timesteps.append(update*nbatch)
            evs.append(float(ev))    
      
    env.close()
    
    # simple logging
    logfile = open('log_timesteps.txt', 'w')
    for t in timesteps:
      logfile.write("%s\n" % t)
    logfile.close()
    logfile2 = open('log_explained_variance.txt', 'w')
    for ev in evs:
      logfile2.write("%s\n" % ev)
    logfile2.close()
    
    return model
