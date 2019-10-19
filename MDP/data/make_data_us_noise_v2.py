import pandas as pd
import numpy as np
import random

act_2_actnum_dict = {"greeting": 0, "end": 1, "up_yes": 2, "up_no": 3, "up_q": 4, "down_disa": 5, "down_ans": 6}
actnum_2_act_dict = {0:"greeting", 1:"end", 2:"up_yes", 3:"up_no", 4:"up_q", 5:"down_disa", 6:"down_ans"}

state_dict = {"[4,1,0]": "down_disa", "[4,2,0]": "down_ans", "[1,3,0]": "up_yes", "[2,3,0]": "up_no", "[3,3,0]": "up_q",
              "[0,0,0]": "greeting",
              "[4,1,1]": "end", "[4,2,1]": "end", "[1,3,1]": "end", "[2,3,1]": "end","[3,3,1]": "end", "[0,0,1]": "end"}


# 全部都空出0.06,后面再拼接上就好
new_state_dict = {"[4,1,0]": [["[4,1,0]","[4,2,0]","[4,1,1]"],[0.25, 0.65, 0.04]],
                  "[4,2,0]": [["[4,2,1]","[4,2,0]","[4,1,0]"],[0.53, 0.38, 0.03]],
                  "[1,3,0]": [["[1,3,1]","[1,3,0]","[2,3,0]","[3,3,0]","[4,1,0]","[4,2,0]"],[0.49,0.04,0.04,0.14,0.09,0.14]],
                  "[2,3,0]": [["[1,3,1]","[1,3,0]","[2,3,0]","[3,3,0]","[4,1,0]","[4,2,0]"],[0.49,0.04,0.04,0.14,0.09,0.14]],
                  "[3,3,0]": [["[2,3,0]","[3,3,0]","[3,3,1]"],[0.45,0.45,0.04]],
                  "[0,0,0]": [["[4,1,0]","[4,2,0]","[3,3,0]","[0,0,1]"],[0.1132, 0.4318, 0.355, 0.04]]}

# "[2,2,1]",2,[],-30,1,2 so it works




df_rewards = pd.read_csv("rewards.csv")
reward_dict = {}
for i in range(len(df_rewards)):
    st = df_rewards["s_old"][i] + str(act_2_actnum_dict[df_rewards["action"][i]])
    if st in reward_dict:
        print("error")
    else:
        reward_dict[st] = int(df_rewards["reward"][i])


# 以一定的概率选取元素 [1,2,3,4], [0.1, 0.2, 0.3, 0.4]
def random_pick(some_list,probabilities):
    x=random.uniform(0,1)
    cumulative_probability=0.0
    for item,item_probability in zip(some_list,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability: break
    return item

def str_2_list(str):
    str = str[1:-1].split(",")
    return [int(s) for s in str]


def make_us():
    X = pd.DataFrame()
    state = []
    action = []
    new_state = []
    reward = []
    terminal = []
    dia_id = []

    for k in range(10000):
        count = 0
        s = "[0,0,0]"
        while 1:
            dia_id.append(count)
            state.append(s)
            if s in state_dict:
                action.append(state_dict[s])
                reward.append(reward_dict[s + str(act_2_actnum_dict[state_dict[s]])])
            else:
                # impossible state -> random action
                a = random.randint(0,6)
                action.append(actnum_2_act_dict[a])
                reward.append(reward_dict[s + str(a)])


            if s[-2] == "1":
                new_state.append([])
                terminal.append(1)
                break
            terminal.append(0)

            # noise version below sentence
            if count >= 9 or s not in new_state_dict:
                # 防止对话过长，强制停下来
                st = ""
                for i in range(len(s)):
                    if i == len(s)-2:
                        st += "1"
                    else:
                        st += s[i]
                new_s = st

            else:
                noise = str([random.randint(0,4), random.randint(0,3), random.randint(0,1)])
                # [2, 1, 1] -> [2,1,1]
                noise = noise.replace(" ", "")
                noise_state = [x for x in new_state_dict[s][0]]
                noise_state.append(noise)
                p = [x for x in new_state_dict[s][1]]
                p.append(0.06)
                new_s = random_pick(noise_state, p)

            new_state.append(new_s)

            s = new_s
            count += 1

    X["state"] = state
    X["action"] = action
    X["new_state"] = new_state
    X["reward"] = reward
    X["terminal"] = terminal
    X["dia_id"] = dia_id

    X.to_csv("train_data.csv", index=False)


if __name__ == '__main__':
    make_us()