import random

class UserSimulator:
    def __init__(self):
        self.possible = [[0,0,0],[5,1,0],[5,2,0],[1,3,0],[2,3,0],[3,3,0],[4,3,0]]
        # self.state_dict = {"[4,1,0]": "down_disa", "[4,2,0]": "down_ans", "[1,3,0]": "up_yes", "[2,3,0]": "up_no",
        #               "[3,3,0]": "up_q",
        #               "[0,0,0]": "greeting",
        #               "[4,1,1]": "end", "[4,2,1]": "end", "[1,3,1]": "end", "[2,3,1]": "end", "[3,3,1]": "end",
        #               "[0,0,1]": "end"}

        self.act_2_actnum_dict = {"greeting": 0, "end": 1, "up_yes": 2, "up_no": 3, "up_q": 4, "down_disa": 5, "down_ans": 6}

        self.new_state_dict_no_noise = {"[5,1,0]_5": [["[5,1,0]", "[5,2,0]"], [0.3, 0.7]],
                                   "[5,2,0]_6": [["[5,2,1]", "[5,2,0]", "[5,1,0]","[1,3,0]","[2,3,0]","[3,3,0]","[4,3,0]"],
                                                 [0.4, 0.25, 0.25,0.025,0.025,0.025,0.025]],
                                   "[1,3,0]_2": [["[1,3,1]", "[1,3,0]", "[2,3,0]", "[3,3,0]", "[4,3,0]","[5,1,0]", "[5,2,0]"],
                                               [0.3, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05]],
                                   "[2,3,0]_3": [["[1,3,1]", "[1,3,0]", "[2,3,0]", "[3,3,0]", "[4,3,0]","[5,1,0]", "[5,2,0]"],
                                                 [0.3, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05]],
                                   "[3,3,0]_4": [["[1,3,0]", "[2,3,0]"], [0.5, 0.5]],
                                   "[4,3,0]_5": [["[4,3,0]","[1,3,0]", "[2,3,0]","[3,3,0]"], [0.1, 0.3, 0.3, 0.3]],
                                   "[0,0,0]_0": [["[5,1,0]", "[5,2,0]", "[1,3,0]","[2,3,0]","[3,3,0]","[4,3,0]", "[0,0,1]"],
                                               [0.25, 0.25, 0.12, 0.12, 0.12,0.12,0.02]]}


        # 以一定的概率选取元素 [1,2,3,4], [0.1, 0.2, 0.3, 0.4]
    def random_pick(self, some_list, probabilities):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(some_list, probabilities):
            cumulative_probability += item_probability
            if x < cumulative_probability: break
        return item

    def user_step(self, state, action):
        state_action_str = str(state).replace(" ", "")+"_"+str(action)

        # 这边已经没有terminal的情况了； 对的状态——最优动作
        if state_action_str in self.new_state_dict_no_noise:
            new_state_str = self.random_pick(self.new_state_dict_no_noise[state_action_str][0],
                                         self.new_state_dict_no_noise[state_action_str][1])

            # new_state_str -> state_list
            new_state_str = new_state_str[1:-1].split(",")
            new_state = [int(s) for s in new_state_str]

        else:
            # 对的状态——非最优动作 + 不可能的状态
            new_state = [random.randint(0,5), random.randint(0,3), random.randint(0,1)]
            while new_state in self.possible:
                new_state = [random.randint(0, 5), random.randint(0, 3), random.randint(0, 1)]

        return new_state