import numpy as np
import pandas as pd
import itertools
import random
import us_goal
import draw
import copy

us = us_goal.UserSimulator()

act_2_actnum_dict = {"greeting": 0, "end": 1, "up_yes": 2, "up_no": 3, "up_q": 4, "down_disa": 5, "down_ans": 6}


df_rewards = pd.read_csv("../data/rewards.csv")
reward_dict = {}
for i in range(len(df_rewards)):
    st = df_rewards["s_old"][i] + str(act_2_actnum_dict[df_rewards["action"][i]])
    if st in reward_dict:
        print("error")
    else:
        reward_dict[st] = int(df_rewards["reward"][i])

# right = [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1),
#          (13, 1), (14, 2), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 3), (23, 1), (24, 1),
#          (25, 1), (26, 1), (27, 1), (28, 1), (29, 1), (30, 4), (31, 1), (32, 1), (33, 1), (34, 5), (35, 1), (36, 6),
#          (37, 1), (38, 1), (39,1)]
#  # right = [(0, 0), (1, 1), (14, 2),  (22, 3), (30, 4), (34, 5), (36, 6)]
# right_dict = {}
# for r in right:
#     right_dict[r[0]] = r[1]
# assert len(right_dict) == 40

class Train():
    def __init__(self):
        self.stateNum = 48
        self.actNum = 7
        self.Q = np.zeros([self.stateNum, self.actNum])
        self.DISCOUNT_FACTOR = 0.9

    # def precision_for_seven_right(self):
    #     for r in range(len(self.Q)):
    #         for j in range(len(self.Q[r])):
    #             self.Q[r][j] = int(self.Q[r][j])
    #
    #     pos = 0
    #     for k, v in right_dict.items():
    #         if np.argmax(self.Q[k]) == v:
    #             pos += 1
    #     return pos/len(right_dict)


    # def precision(self):
    #     for r in range(len(self.Q)):
    #         for j in range(len(self.Q[r])):
    #             self.Q[r][j] = int(self.Q[r][j])
    #
    #     res = []
    #
    #     for i in range(len(self.Q)):
    #         # if max(self.Q[i]) != 0:
    #         res.append((i,np.argmax(self.Q[i])))
    #
    #     pos = 0
    #     all = 0
    #     for r in res:
    #         if r[0] in right_dict:
    #             if r[1] == right_dict[r[0]]:
    #                 pos += 1
    #             all += 1
    #     print(self.Q)
    #     print(pos)
    #     print(all)
    #
    #     return pos / all

    def state_2_statenum(self, state):
        # state [0,0,1]  = 0*5^2 + 0*4^1 + 1*2^0 = 1
        return state[0]*8 + state[1]*2 + state[2] # 顺序不能乱 从右到左 且 8 = 2^3 我也不知道这是为什么

    def policy(self, Q, state, epsilon=0.1):
        action_probs = np.ones(self.actNum, dtype=float) * epsilon / self.actNum
        # 下面这句换成根据Q表来的
        # act_num = act_2_actnum_dict[self.trainData['action'][i_now]] # 将动作转换为起对应的数字 clarify--0 confirm--1 answer--2
        act_num = np.argmax(Q[state])
        action_probs[act_num] += (1.0 - epsilon)
        action = np.random.choice(self.actNum, p=action_probs)
        return action

    def step(self, state, action, step_num):
        state_action_str = str(state).replace(" ", "") + str(action)
        terminal = 0
        reward = reward_dict[state_action_str]

        # step_num > 25 防止对话长度过长,强制停下来
        # TODO 长度过长应该给惩罚
        if step_num > 40:
            terminal = 1
            next_state = []
            reward = -20

        elif state[2] == 1 or action == 1:
            terminal = 1
            next_state = []

        else:
            # 这里相当于用户模拟器的部分
            next_state = us.user_step(state, action)

        return next_state, reward, terminal

    def q_learning(self, num_episodes=10000, alpha=0.5, epsilon=0.3, span=100):
        reward_log_epi_list = []
        metrics = []
        check_R = 0
        count = 0
        check = []

        metrics_pre = []
        episode = []

        Q_record = []

        t_reward_list = []
        t_check_if_suc_flag = 0
        t_check_if_suc_list = []

        for ith in range(1, num_episodes + 1):  # 大循环
            reward_log_turn_list = []
            count += 1
            state_list = [0,0,0]
            R = 0
            t_reward = 0
           
            for t in itertools.count():  # 小循环
                state_num = self.state_2_statenum(state_list)
                action = self.policy(self.Q, state_num, epsilon)
                next_state_list, reward, terminal= self.step(state_list, action, t)

                reward_log_turn_list.append(reward)

                check_R += reward
                check.append(check_R/count)

                if reward > 0:
                    R += 1
                else:
                    R += -1

                t_reward += 1
                if reward == 20:
                    t_reward_list.append(t_reward)
                    t_reward = 0
                    t_check_if_suc_flag = 1

                if terminal:
                    self.Q[state_num][action] = reward
                    # self.Q[state][action] = reward + self.Q[state][action]
                    break
                else:
                    next_state_num = self.state_2_statenum(next_state_list)
                    best_action = np.argmax(self.Q[next_state_num])
                    td_target = reward + self.DISCOUNT_FACTOR * self.Q[next_state_num][best_action]
                    self.Q[state_num][action] = self.Q[state_num][action] + alpha * (td_target - self.Q[state_num][action])
                    state_list = next_state_list
                    # step_num += 1
            if t_check_if_suc_flag:
                t_check_if_suc_list.append(1)
                t_check_if_suc_flag = 0
            else:
                t_check_if_suc_list.append(0)

            metrics.append(R / (t + 1))

            if ith % 100 == 0:
                # metrics_pre.append(self.precision_for_seven_right())
                # metrics_pre.append(self.precision())
                episode.append(num_episodes)
                Q_record.append(copy.deepcopy(self.Q))

            reward_log_epi_list.append(reward_log_turn_list[:])
            # for end

        draw.process(metrics_pre)


        np.save('../data/Q_Table.npy', self.Q)
        print("Q_Table has saved...")
        print(self.Q)
        for i, x in enumerate(self.Q):
            print (i, np.argmax(x))


        Q_diff = []
        for i in range(len(Q_record)-1):
            Q_diff.append(np.mean(Q_record[i+1] - Q_record[i]))

        # for reward_log
        reward_log_epi_res = []
        for turn_list in reward_log_epi_list:
            turn_list = turn_list[::-1]
            if 10 in turn_list:
                p = turn_list.index(10)
                res = turn_list[:p].count(-1) * 10 + (turn_list[p:].count(-1) / turn_list[p:].count(10))
            else:
                # just big number given
                res = 100
            reward_log_epi_res.append(res)

        # print(reward_log_epi_res)
        # draw.process(reward_log_epi_res)
        for_draw_reward_log_epi_res = []
        for i in range(int(num_episodes / span)):
            for_draw_reward_log_epi_res.append(sum(reward_log_epi_res[span * i:span * (i + 1)])/span)

        # print(for_draw_reward_log_epi_res)
        #
        # draw.process(for_draw_reward_log_epi_res)
        return for_draw_reward_log_epi_res



        # 画图
        # t_check_if_suc_draw = []
        # for i in range(int(num_episodes/span)):
        #     t_check_if_suc_draw.append(sum(t_check_if_suc_list[span*i:span*(i+1)]))
        #
        # # print(t_check_if_suc_draw)
        # f.close()
        # return t_check_if_suc_draw
        # draw.process(t_check_if_suc_draw)


        # draw.process(t_check_if_suc_list)
        # draw.process(t_reward_list)

        # draw.process(Q_diff)
        # draw.process(metrics)



    def q_learning_test(self, num_episodes=10000, epsilon=0, span=100):
        # self.Q = np.load('data/Q_Table_done.npy')
        self.Q = np.load('data/Q_Table.npy')
        reward_log_epi_list = []

        t_reward_list = []
        t_check_if_suc_flag = 0
        t_check_if_suc_list = []
        for ith in range(1, num_episodes + 1):  # 大循环
            reward_log_turn_list = []
            state_list = [0, 0, 0]

            if t_check_if_suc_flag:
                t_check_if_suc_list.append(1)
                t_check_if_suc_flag = 0
            else:
                t_check_if_suc_list.append(0)
            t_reward = 0

            for t in itertools.count():  # 小循环
                state_num = self.state_2_statenum(state_list)
                action = self.policy(self.Q, state_num, epsilon)
                next_state_list, reward, terminal = self.step(state_list, action, t)
                reward_log_turn_list.append(reward)

                t_reward += 1
                if reward == 20:
                    t_reward_list.append(t_reward)
                    t_reward = 0
                    t_check_if_suc_flag = 1

                if terminal:
                    break
                else:
                    state_list = next_state_list
            reward_log_epi_list.append(reward_log_turn_list)

        reward_log_epi_res = []
        for turn_list in reward_log_epi_list:
            turn_list = turn_list[::-1]
            if 10 in turn_list:
                p = turn_list.index(10)
                res = turn_list[:p].count(-1) * 10 + (turn_list[p:].count(-1) / turn_list[p:].count(10))
            else:
                # just big number given
                res = 100
            reward_log_epi_res.append(res)

        for_draw_reward_log_epi_res = []
        for i in range(int(num_episodes / span)):
            for_draw_reward_log_epi_res.append(sum(reward_log_epi_res[span * i:span * (i + 1)]) / span)

        # print(reward_log_epi_res)
        # draw.process(reward_log_epi_res)

        # 画图
        # t_check_if_suc_draw = []
        #
        # for i in range(int(num_episodes/span)):
        #     t_check_if_suc_draw.append(sum(t_check_if_suc_list[span * i:span * (i + 1)])/span)

        return for_draw_reward_log_epi_res


if __name__ == '__main__':
    tr = Train()

    train_num_episodes = 5000
    text_num_episodes = 300
    tr.q_learning(num_episodes=train_num_episodes, span=100)
    # tr.q_learning_test(num_episodes=text_num_episodes, epsilon=0)

    # for 成功过
    # train
    # train_num_episodes = 10000
    # text_num_episodes = 4000
    # num = 1
    # span = 100
    # res = [0]*int(train_num_episodes/span)
    # res_test = [0] * int(text_num_episodes/span)
    # for i in range(num):
    #     res = np.sum([res, tr.q_learning(num_episodes=train_num_episodes, span=span)], axis=0)
    #     res_test = np.sum([res_test, tr.q_learning_test(num_episodes=text_num_episodes, epsilon=0, span=span)], axis=0)
    #     print(i, "done...")
    #
    # res = list(np.array(res)/num)
    # res_test = list(np.array(res_test) / num)
    #
    # res.extend(res_test)
    #
    # # test
    # draw.process(res)




