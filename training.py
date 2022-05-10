import matplotlib.pyplot as plt
import numpy as np
from anesthetic import MCMCSamples
import pypolychord as pp
from pypolychord.settings import PolyChordSettings as Settings
import tensorflow as tf
from keras import layers
import re

plt.switch_backend("TkAgg")

# Load the planck samples into MCMCSamples object
root = 'base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing'
planck_samples = MCMCSamples(root=root)
# -------------------------------------Start of training script-------------------------


# Specific parameters for a polychord sampling process
n_live = 50
nDims = 6

# Defining problem space
# state: nlive,ndead,mu,sig
num_states = nDims * (nDims + 1) + 2
num_actions = nDims
upper_bound = np.ones(nDims) * 5
lower_bound = np.zeros(nDims)


# Initializing action noises
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    outputs = (1 + outputs) / 2
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


# ---------------------Setting Training hyper parameters---------------------------
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# Defining actor and critic models
actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()
# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.005
actor_lr = 0.01

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 10
# Discount factor for future rewards
gamma = 0.7
# Used to update target networks
tau = 0.01

buffer = Buffer(1000, 10)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


class PolychordProcess:
    def __init__(self, nDims, n_live, buffer_object):
        self.serial_num = -1
        self.nDims = nDims
        self.nlive = n_live
        self.nDerived = 0
        self.paramnames = planck_samples.columns[:self.nDims]
        self.mu = planck_samples[self.paramnames].mean().values
        self.sig = planck_samples[self.paramnames].cov().values
        self.invSig = np.linalg.inv(self.sig)
        self.bounds = np.array([planck_samples.limits[p] for p in self.paramnames], dtype=float)
        self.upper = self.bounds[:, 1]
        self.lower = self.bounds[:, 0]
        self.diff = self.upper - self.lower
        self.vol_og = np.prod(self.diff)
        self.action = None
        self.buffer = buffer_object
        # ------------------Mode 1: Individual Scaling---------------------------------------
        # state = np.zeros(2 * self.nDims + 7)
        # state[:self.nDims] = 1e10 * np.ones(self.nDims)
        # state[self.nDims:self.nDims + 2] = -1e20 * np.ones(2)
        # state[self.nDims + 2:2 * self.nDims + 2] = 1e10 * np.ones(self.nDims)
        # state[-5:-3] = -1e20 * np.ones(2)
        # state[-3] = -1e20
        # state[-2] = 1e10
        # state[-1] = 0
        # self.state = np.tanh(state)
        # -----------------Mode 2: Total scaling---------------------------------------------
        state = np.zeros(nDims * (nDims + 1) + 2)
        state[:nDims] = self.mu
        state[nDims: nDims * (nDims + 1)] = self.sig.reshape((nDims ** 2,))
        state[-2] = self.nlive
        state[-1] = 0
        self.state = state
        self.episodic_reward = 0

    def reset(self):
        self.serial_num += 1
        self.base_dir = "train/d_{}".format(self.serial_num)
        self.action = None
        # -----------------Mode 1: Individual Scaling---------------------------------------
        # state = np.zeros(2 * self.nDims + 7)
        # state[:self.nDims] = 1e10 * np.ones(self.nDims)
        # state[self.nDims:self.nDims + 2] = -1e20 * np.ones(2)
        # state[self.nDims + 2:2 * self.nDims + 2] = 1e10 * np.ones(self.nDims)
        # state[-5:-3] = -1e20 * np.ones(2)
        # state[-3] = -1e20
        # state[-2] = 1e10
        # state[-1] = 0
        # self.state = np.tanh(state)
        # -----------------Mode 2: Total scaling---------------------------------------------
        state = np.zeros(nDims * (nDims + 1) + 2)
        state[:nDims] = self.mu
        state[nDims: nDims * (nDims + 1)] = self.sig.reshape((nDims ** 2,))
        state[-2] = self.nlive
        state[-1] = 0
        self.state = state
        self.episodic_reward = 0

    def ppf(self, cube_entry):
        return cube_entry * self.diff + self.lower

    def cdf(self, theta_entry):
        return (theta_entry - self.lower) / self.diff

    def update_bounds(self):
        # -------Incremental mode------------
        # beta += self.action[0]
        # beta_width += self.action[1]
        # beta = np.clip(beta, 0.01, 1)
        # beta_width = np.clip(beta_width, 1, 5)
        upper_new = self.mu + self.action * np.diag(self.sig)
        lower_new = self.mu - self.action * np.diag(self.sig)
        self.x_upper = np.clip(self.cdf(upper_new), 0, 1)
        self.x_lower = np.clip(self.cdf(lower_new), 0, 1)
        self.upper_new = self.ppf(self.x_upper)
        self.lower_new = self.ppf(self.x_lower)
        self.x_diff = self.x_upper - self.x_lower
        self.diff_new = self.upper_new - self.lower_new

    # # Function that communicates with the RL training algorithm to produce current state vector
    # def dumper(self, live_points, dead_points, logweights, logZ, logZstd):
    #     with open('{}/default.stats'.format(self.base_dir), 'r') as f:
    #         for _ in range(15):
    #             line = f.readline()
    #         logZs = []
    #         logZerrs = []
    #         while line[:5] == 'log(Z':
    #             logZs.append(float(re.findall(r'=(.*)', line
    #                                           )[0].split()[0]))
    #             logZerrs.append(float(re.findall(r'=(.*)', line
    #                                              )[0].split()[2]))
    # 
    #             line = f.readline()
    # 
    #         for _ in range(5):
    #             f.readline()
    # 
    #         ncluster = len(logZs)
    #         nposterior = int(f.readline().split()[1])
    #         nequals = int(f.readline().split()[1])
    #         self.n_dead = int(f.readline().split()[1])
    #         nlive = int(f.readline().split()[1])
    #         # Protect against ValueError when .stats file has ******* for nlike
    #         # (occurs when nlike is too big).
    #         try:
    #             nlike = int(f.readline().split()[1])
    #         except ValueError:
    #             nlike = 1e30
    #         logX = float(f.readline().split()[1])
    #         line = f.readline()
    #         line = line.split()
    #         i = line.index('(')
    #         avnlike = [float(x) for x in line[1:i]]
    # 
    #     # -------------------------------------Mode 1: Individual scaling,
    #     # # Generating new state from polychord output function after 100 new live points have been generated
    #     # new_state = np.empty_like(self.state)
    #     # new_state[:self.nDims] = np.sqrt(np.sum(np.square(np.subtract(live_points[:, :27], self.mu)), axis=0))
    #     # new_state[self.nDims:self.nDims + 2] = np.sum(live_points[:, -2:], axis=0)
    #     # new_state[self.nDims + 2:2 * self.nDims + 2] = np.sqrt(
    #     #     np.sum(np.square(np.subtract(dead_points[:, :27], self.mu)), axis=0))
    #     # new_state[-5:-3] = np.sum(dead_points[:, -2:], axis=0)
    #     # new_state[-3] = logZ
    #     # new_state[-2] = logZstd
    #     # new_state[-1] = self.n_dead
    #     # # Encoding the new state with tanh function
    #     # new_state = np.tanh(new_state)
    #     # -----------------Mode 2: Total scaling---------------------------------------------
    #     new_state = np.array([ncluster, nposterior, nequals, self.n_dead, nlive, nlike, avnlike[0]])
    #     # Reward Function
    #     logZ_live = np.log(np.sum(np.exp(live_points[:, -1])) / nlive) + logX
    #     if np.isnan(logZ_live) or (logZ_live <= np.log(0.001) + logZ and not np.isinf(logZ_live)):
    #         reward = 100 - 10 * np.abs(logZ - right_answer)
    #         # Finish line!
    #     elif avnlike[0] > 1000:
    #         reward = -101 - 100 * (ncluster - 1)
    #     else:
    #         reward = -1 - 100 * (ncluster - 1)
    # 
    #     print(reward)
    #     # Recording transition tuple into Buffer
    #     self.buffer.record((self.state, self.action, reward, new_state))
    # 
    #     # Calculating next action
    #     tf_prev_state = tf.expand_dims(tf.convert_to_tensor(new_state), 0)
    #     self.action = policy(tf_prev_state, ou_noise)[0]
    #     self.action_list.append(self.action)
    # 
    #     # Updating new prior proposal bounds
    #     # self.update_bounds()
    #     beta = self.action
    #     beta_list.append(beta)
    #     self.counter += 1
    #     # Updating State
    #     self.state = new_state
    #     self.logZ = logZ
    # 
    #     # Accumulating episodic reward
    #     self.episodic_reward += reward
    # 
    #     # Learning
    #     buffer.learn()
    #     update_target(target_actor.variables, actor_model.variables, tau)
    #     update_target(target_critic.variables, critic_model.variables, tau)

    def run(self):
        self.update_bounds()

        def loglikelihood(theta):
            return -(theta - self.mu) @ self.invSig @ (theta - self.mu) / 2, []

        def loglikelihood_tilde(theta):
            beta = theta[0]
            theta = theta[1:]
            return loglikelihood(theta)[0] + np.log(1 / (self.vol_og * pi_tilde(theta, beta))), []

        def pi_tilde(theta, beta):
            return np.product(
                beta / self.diff + (1 - beta) * ((theta < self.upper_new) & (theta >= self.lower_new)) / self.diff_new)

        def prior(cube_input):
            # When beta = 0, new proposal only, again, a uniform distribution but with bounds being proposal
            # prior bounds
            # beta = 1
            beta = cube_input[0]
            theta = np.empty_like(cube_input)
            cube_input = cube_input[1:]
            if beta == 0:
                cube = cube_input
                theta[1:] = cube * self.x_diff + self.x_lower
            else:
                cube = (cube_input < beta * self.x_lower) * (cube_input / beta) + (
                        (cube_input >= beta * self.x_lower) & (
                        cube_input < beta * self.x_upper + (1 - beta))) * (
                               cube_input + (1 - beta) * self.x_lower / self.x_diff) / (
                               beta + (1 - beta) / self.x_diff) + (
                               cube_input >= (beta * self.x_upper + (1 - beta))) * (
                               cube_input - (1 - beta)) / beta
                theta[1:] = self.ppf(cube)
            theta[0] = beta
            return theta

        settings = Settings(self.nDims + 1, self.nDerived, file_root="default",
                            nlive=self.nlive,
                            write_prior=False,
                            read_resume=False,
                            write_resume=False,
                            base_dir=self.base_dir,
                            do_clustering=False
                            )

        pp.run_polychord(loglikelihood_tilde, self.nDims + 1, self.nDerived, settings, prior=prior)
        with open('{}/default.stats'.format(self.base_dir), 'r') as f:
            for _ in range(15):
                line = f.readline()
            logZs = []
            logZerrs = []
            while line[:5] == 'log(Z':
                logZs.append(float(re.findall(r'=(.*)', line
                                              )[0].split()[0]))
                logZerrs.append(float(re.findall(r'=(.*)', line
                                                 )[0].split()[2]))

                line = f.readline()

            for _ in range(7):
                f.readline()
            n_dead = int(f.readline().split()[1])
        reward = -n_dead/n_live
        new_state = self.state
        new_state[-1] = n_dead

        # Recording transition tuple into Buffer
        self.buffer.record((self.state, self.action, reward, new_state))

        # Calculating next action
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(new_state), 0)

        # Accumulating episodic reward
        self.episodic_reward += reward

        # Learning
        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)


# Initializing a new PolychordProcess object for all polychord runs
env = PolychordProcess(n_live=n_live, nDims=nDims, buffer_object=buffer)
# -------------------------Training Begins-----------------------------------
for ep in range(total_episodes):
    env.reset()
    episodic_reward = 0
    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(env.state), 0)
    env.action = policy(tf_prev_state, ou_noise)[0]
    env.run()
    ep_reward_list.append(env.episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-5:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    # x = np.arange(env.counter + 1)
    # plt.plot(x, env.action_list, label="action")
    # plt.xlabel("Iteration")
    # plt.ylabel("Progress")
    # plt.plot(x, env.beta_list, label="beta")
    # plt.legend()
    # plt.title("Action and beta against iterations in one sampling run")
    # plt.show()

    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

actor_model.save_weights("actor_weights.h5")
critic_model.save_weights("critic_weights.h5")

target_actor.save_weights("actor_target_weights.h5")
target_critic.save_weights("critic_target_weights.h5")
