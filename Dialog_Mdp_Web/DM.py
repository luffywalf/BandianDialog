import numpy as np
class DM:
    def __init__(self):
        self.Q_path = "src/generate/Q_Table.npy"
        self.Q = np.load(self.Q_path)
        # self.act_2_actnum_dict = {"greeting": 0, "end": 1, "up_yes": 2, "up_no": 3, "up_q": 4, "down_disa": 5, "down_ans": 6}

    def state_2_statenum(self, state):
        # state [0,0,1]  = 0*5^2 + 0*4^1 + 1*2^0 = 1
        return state[0]*8 + state[1]*2 + state[2] # 顺序不能乱 从右到左 且 8 = 2^3 我也不知道这是为什么

    def get_action(self, mdp_state):
        state_num = self.state_2_statenum(mdp_state)
        if state_num >= len(self.Q):
            print("error: state_num")
            action_num = 0
        else:
            action_num = np.argmax(self.Q[state_num])
        return action_num





