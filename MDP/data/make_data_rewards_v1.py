import pandas as pd
import itertools
'''
这里面暂时还没有体现时序的槽值
'''

X = pd.DataFrame()
a = [0,1,2,3,4,5]
b = [0,1,2,3]
c = [0,1]

actions = ["greeting","end","up_q","up_yes","up_no","down_disa","down_ans"]
res = list(itertools.product(a,b,c,actions))

for i, x in enumerate(itertools.product(a,b,c)):
    print(i, x)

s_old = []
action = []
reward = []


with open('rewards.txt', 'wt') as f:
    for x in res:
        # x[0]-up  x[1]-down  x[2]-terminal  x[3]-actions

        # greeting
        if x[0] == x[1] == x[2] == 0:
            # 选对动作
            if x[3] == "greeting":
                s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                action.append(x[3])
                reward.append(-1)
                f.write(str(x) + ", -1" + "\n")
            else:
                s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                action.append(x[3])
                reward.append(-1)
                f.write(str(x) + ", -1" + "\n")

        # end
        elif x[2] == 1:
            if x[3] == "end":
                s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                action.append(x[3])
                reward.append(-1)
                f.write(str(x) + ", -1" + "\n")
            else:
                s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                action.append(x[3])
                reward.append(-1)
                f.write(str(x) + ", -1" + "\n")

        # up, intent == 0 条件类
        # 终止为0
        elif x[0] != 5 and x[1] == 3:
            if x[0] == 1:
                if x[3] == "up_yes":
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(10)
                    f.write(str(x) + ", +10" + "\n")
                else:
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(-1)
                    f.write(str(x) + ", -1" + "\n")
            elif x[0] == 2:
                if x[3] == "up_no":
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(10)
                    f.write(str(x) + ", +10" + "\n")
                else:
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(-1)
                    f.write(str(x) + ", -1" + "\n")
            elif x[0] == 3:
                if x[3] =="up_q":
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(-1)
                    f.write(str(x) + ", -1" + "\n")
                else:
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(-1)
                    f.write(str(x) + ", -1" + "\n")
            elif x[0] == 4:
                if x[3] =="down_disa":
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(-1)
                    f.write(str(x) + ", -1" + "\n")
                else:
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(-1)
                    f.write(str(x) + ", -1" + "\n")
            elif x[3] == "end":
                s_old.append(str([x[0], x[1], x[2]]).replace(" ", ""))
                action.append(x[3])
                reward.append(-1)
                f.write(str(x) + ", -1" + "\n")
            else:
                # no such state
                s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                action.append(x[3])
                reward.append(-1)
                f.write(str(x) + ", -1" + "\n")

        # down, intent == 1 含义类
        elif x[0] == 5 and x[1] != 3:
            # disambiguation
            if x[1] == 1:
                if x[3] == "down_disa":
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(-1)
                    f.write(str(x) + ", -1" + "\n")
                else:
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(-1)
                    f.write(str(x) + ", -1" + "\n")
            # answer
            elif x[1] == 2:
                if x[3] == "down_ans":
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(10)
                    f.write(str(x) + ", +10" + "\n")
                else:
                    s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                    action.append(x[3])
                    reward.append(-1)
                    f.write(str(x) + ", -1" + "\n")
            elif x[3] == "end":
                s_old.append(str([x[0], x[1], x[2]]).replace(" ", ""))
                action.append(x[3])
                reward.append(-1)
                f.write(str(x) + ", -1" + "\n")
            else:
                # no such state
                s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
                action.append(x[3])
                reward.append(-1)
                f.write(str(x) + ", -1" + "\n")
        elif x[3] == "end":
            s_old.append(str([x[0], x[1], x[2]]).replace(" ", ""))
            action.append(x[3])
            reward.append(-1)
            f.write(str(x) + ", -1" + "\n")
        else:
            # no such state
            s_old.append(str([x[0],x[1],x[2]]).replace(" ",""))
            action.append(x[3])
            reward.append(-1)
            f.write(str(x) + ", -1" + "\n")


X = pd.DataFrame()
X["s_old"] = s_old
X["action"] = action
X["reward"] = reward

X.to_csv("rewards.csv", index=False)

print(len(res))

