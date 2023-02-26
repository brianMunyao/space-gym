import torch as T
import numpy as np

from DQN import DQN


class Agent:
    def __init__(self, gamma, eps, LR, max_mem_size, eps_end=0.5, replace=5000, action_space=[0, 1, 2, 3, 4, 5]):
        self.gamma = gamma
        self.eps = eps
        self.eps_end = eps_end
        self.action_space = action_space
        self.max_mem_size = max_mem_size
        self.steps = 0
        self.learn_step_counter = 0  # for target network replacement
        self.memory = []  # use as a list
        self.mem_counter = 0
        self.replace_target_counter = replace
        self.Q_eval = DQN(LR)  # agent's estimate of the current set of states
        # agent's estimate of the successor set of states
        self.Q_next = DQN(LR)

    def store_transition(self, state, action, reward, resulting_state):
        if self.mem_counter < self.max_mem_size:
            self.memory.append([state, action, reward, resulting_state])
        else:
            self.memory[self.mem_counter % self.max_mem_size] = [
                state, action, reward, resulting_state]

        self.mem_counter += 1

    def choose_action(self, observation):
        # we pass in a sequence of observations
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.eps:
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.action_space)

        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()  # batch learning, zero grad
        if self.replace_target_counter is not None and self.learn_step_counter % self.replace_target_counter == 0:
            self.Q_next.load_state_dict(self.Q_eval.state.dict())

        if self.mem_counter + batch_size < self.max_mem_size:
            mem_start = int(np.random.choice(range(self.mem_counter)))
        else:
            mem_start = int(np.random.choice(
                range(self.max_mem_size-batch_size-1)))

        mini_batch = self.memory[mem_start:mem_start+batch_size]
        memory = np.array(mini_batch)

        Qpred = self.Q_eval.forward(
            list(memory[:, 0][:])).to(self.Q_eval.device)
        Qnext = self.Q_next.forward(
            list(memory[:, 3][:])).to(self.Q_eval.device)

        maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = T.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
        Qtarget = Qpred
        Qtarget[:, maxA] = rewards + self.gamma*T.max(Qnext[1])

        if self.steps > 500:
            if self.eps - 1e-4 > self.eps_end:
                self.eps -= 1e-4  # converge epsilon
        else:
            self.eps = self.eps_end

        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
