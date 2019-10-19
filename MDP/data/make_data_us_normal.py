import pandas as pd
import random

state_dict = {"[4,1,0]": "down_disa", "[4,2,0]": "down_ans", "[1,3,0]": "up_yes", "[2,3,0]": "up_no", "[3,3,0]": "up_q",
              "[0,0,0]": "greeting",
              "[4,1,1]": "end", "[4,2,1]": "end", "[1,3,1]": "end", "[2,3,1]": "end","[3,3,1]": "end", "[0,0,1]": "end"}


new_state_dict = {"[4,1,0]": [["[4,1,0]","[4,2,0]"],[0.3, 0.7]],
                  "[4,2,0]": [["[4,2,1]","[4,2,0]","[4,1,0]"],[0.55, 0.35, 0.05]],
                  "[1,3,0]": [["[1,3,1]","[1,3,0]","[2,3,0]","[3,3,0]","[4,1,0]","[4,2,0]"],[0.5,0.05,0.05,0.15,0.1,0.15]],
                  "[2,3,0]": [["[1,3,1]","[1,3,0]","[2,3,0]","[3,3,0]","[4,1,0]","[4,2,0]"],[0.5,0.05,0.05,0.15,0.1,0.15]],
                  "[3,3,0]": [["[2,3,0]","[3,3,0]"],[0.5,0.5]],
                  "[0,0,0]": [["[4,1,0]","[4,2,0]","[3,3,0]","[0,0,1]"],[0.1152, 0.4608, 0.384, 0.04]]}
                  # [4, 1, 1]: [], [4, 2, 1]: [], [1, 3, 1]: [], [2, 3, 1]: [], [0, 3, 1]: [],[0, 0, 1]: []

df_rewards = pd.read_csv("rewards.csv")
reward_dict = {}
for i in range(len(df_rewards)):
    str = df_rewards["s_old"][i] + df_rewards["action"][i]
    if str in reward_dict:
        print("error")
    else:
        reward_dict[str] = int(df_rewards["reward"][i])
print(len(reward_dict))


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
            action.append(state_dict[s])
            # rewards 应该都包含了
            reward.append(reward_dict[s+state_dict[s]])

            if s[-2] == "1":
                new_state.append([])
                terminal.append(1)
                break
            terminal.append(0)

            # normal version below sentence
            if count >= 9 and s != '[3,3,0]' and s != '[4,1,0]':
                # 防止对话过长，强制停下来
                str = ""
                for i in range(len(s)):
                    if i == len(s)-2:
                        str += "1"
                    else:
                        str += s[i]
                new_s = str
            else:
                new_s = random_pick(new_state_dict[s][0], new_state_dict[s][1])

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




