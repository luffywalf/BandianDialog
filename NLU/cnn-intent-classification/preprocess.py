import pandas as pd
import random

def make_data_from_bandian_q_pos():
    df_ban_pos = pd.read_csv("/Users/dingning/Ding/Python/VE/BUPT/DialogKB/NLU/resources/database/bandian_q_pos.csv")

    class_dict = {}

    for i in range(len(df_ban_pos)):
        if df_ban_pos["class_label"][i] in class_dict:
            class_dict[df_ban_pos["class_label"][i]].append(df_ban_pos["user_q"][i])

        else:
            class_dict[df_ban_pos["class_label"][i]] = [df_ban_pos["user_q"][i]]

    X = pd.DataFrame()
    q = []

    for k, v in class_dict.items():
        if len(v) <= 5:
            q.extend(v)
        else:
            q.extend(random.sample(v, 5))  # len(v) == 10; take half of it

    cla = [0] * len(q)  # fix label by myself later

    X["query"] = q
    X["class_label"] = cla

    X.to_csv("～", index=False)




if __name__ == '__main__':
    make_data_from_bandian_q_pos()


