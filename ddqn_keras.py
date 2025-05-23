import numpy as np
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras import layers, models, optimizers, losses # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        # Check for NaN values in state, reward, and new_state before storing
        if np.isnan(state).any() or np.isnan(reward) or np.isnan(state_).any():
            print("❌ NaN detected in state or reward, transition not stored.")
            return
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DDQNAgent:
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims,
                 epsilon_dec,replace_target, epsilon_end, mem_size=25000,
                 fname='ddqn_model.keras'):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.alpha = alpha
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)

        self.brain_eval = Brain(input_dims, n_actions, batch_size)
        self.brain_target = Brain(input_dims, n_actions, batch_size)

        # self.optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = np.array(state)
        state = state[np.newaxis, :]
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.brain_eval.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)
        action_indices = np.dot(np.array(actions, dtype=np.int32), np.arange(self.n_actions, dtype=np.int32))

        states = np.array(states, dtype=np.float32)
        states_ = np.array(states_, dtype=np.float32)

        q_next = self.brain_target.predict(states_)
        q_eval_next = self.brain_eval.predict(states_)
        q_pred = self.brain_eval.predict(states)

        # Log Q-values to see if NaNs are present
        # print(f"q_next: {q_next}")
        # print(f"q_eval_next: {q_eval_next}")
        # print(f"q_pred: {q_pred}")

        # --- NaN check ---
        if np.isnan(q_next).any() or np.isnan(q_eval_next).any() or np.isnan(q_pred).any():
            print("❌ NaN detected in Q-value predictions")
            return

        max_actions = np.argmax(q_eval_next, axis=1)

        q_target = q_pred.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = rewards + self.gamma * q_next[batch_index, max_actions] * done # Bellman equation

        # --- NaN check in q_target ---
        if np.isnan(q_target).any():
            print("❌ NaN detected in Q-targets before training")
            return
        
        self.brain_eval.train(states, q_target)

        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    def update_network_parameters(self):
        self.brain_target.copy_weights(self.brain_eval)

    def save_model(self):
        self.brain_eval.model.save(self.model_file)

    def load_model(self):
        self.brain_eval.model = load_model(self.model_file) 
        self.brain_target.model = load_model(self.model_file)

        if self.epsilon == 0.0:
            self.update_network_parameters()


class Brain:
    def __init__(self, NbrStates, NbrActions, batch_size=256):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions
        self.batch_size = batch_size
        self.model = self.createModel()

    def createModel(self):
        model = models.Sequential([
            layers.Input(shape=(self.NbrStates,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.NbrActions, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001,clipvalue=1.0),loss=losses.Huber())
        return model

    def train(self, x, y, epoch=1, verbose=0, optimizer=None):
        if optimizer:
            self.model.compile(optimizer=optimizer, loss=losses.Huber())
        self.model.fit(x, y, batch_size=self.batch_size, epochs=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s, verbose=0)

    def predictOne(self, s):
        s = np.reshape(s, (1, self.NbrStates))
        return self.model.predict(s, verbose=0).flatten()

    def copy_weights(self, TrainNet):
        self.model.set_weights(TrainNet.model.get_weights())
