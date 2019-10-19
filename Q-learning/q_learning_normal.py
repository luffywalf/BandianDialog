import numpy as np
import pandas as pd
import itertools

act_2_actnum_dict = {"greeting": 0, "end": 1, "up_yes": 2, "up_no": 3, "up_q": 4, "down_disa": 5, "down_ans": 6}

df_rewards = pd.read_csv("data/rewards.csv")
reward_dict = {}
for i in range(len(df_rewards)):
    st = df_rewards["s_old"][i] + str(act_2_actnum_dict[df_rewards["action"][i]])
    if st in reward_dict:
        print("error")
    else:
        reward_dict[st] = int(df_rewards["reward"][i])

class Train():
    stateNum = 40
    actNum = 7

    def state_2_statenum(self, state):
        # state [0,0,1]  = 0*5^2 + 0*4^1 + 1*2^0 = 1
        str = state[1:-1].split(",")
        state = [int(s) for s in str]
        return state[0]*8 + state[1]*2 + state[2] # 顺序不能乱 从右到左 且 8 = 2^3 我也不知道这是为什么

    def policy(self, i_now, epsilon=0.1):
        action_probs = np.ones(self.actNum, dtype=float) * epsilon / self.actNum
        act_num = act_2_actnum_dict[self.trainData['action'][i_now]] # 将动作转换为起对应的数字 clarify--0 confirm--1 answer--2
        action_probs[act_num] += (1.0 - epsilon)
        action = np.random.choice(self.actNum, p=action_probs)
        return action

    def q_learning(self):
        discount = 0.8
        lr = 0.9
        self.Q = np.zeros([self.stateNum, self.actNum])
        self.trainData = pd.read_csv("data/train_data.csv")
        item = len(self.trainData)

        i = 0

        while i < item * 1:
            i_now = i % item

            act = self.policy(i_now)
            state = self.state_2_statenum(self.trainData['state'][i_now])
            # r(s,a)
            reward = reward_dict[self.trainData['state'][i_now]+str(act)]
            terminal = self.trainData['terminal'][i_now]

            if terminal:
                self.Q[state][act] = reward

            else:
                next_state = self.state_2_statenum(self.trainData['new_state'][i_now])

                max_Q = reward + discount * max(self.Q[next_state])
                # if max_Q > self.Q[state][act]: res right too...
                self.Q[state][act] = self.Q[state][act] + lr * (max_Q - self.Q[state][act])

            i = i + 1

        for r in range(len(self.Q)):
            for j in range(len(self.Q[r])):
                self.Q[r][j] = int(self.Q[r][j])
        print(self.Q)

        res = []

        for i in range(len(self.Q)):
            if max(self.Q[i]) != 0:
                res.append((i,np.argmax(self.Q[i]) ))

        print(res)
        # [(0, 0), (1, 1), (14, 2), (15, 1), (22, 3), (23, 1), (30, 4), (34, 5), (36, 6), (37, 1)]






if __name__ == '__main__':
    tr = Train()
    tr.q_learning()
    # tr.Q_training_v1()
