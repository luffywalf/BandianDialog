import pandas as pd
import itertools


X = pd.DataFrame()
a = [0,1,2,3]
b = [0,1,2,3]
c = [0,1,2]
d = [0,1]
actions = ["greeting","end","up_q","up_yes","up_no","down_disa","down_user","down_material"]
res = list(itertools.product(a,b,c,d,actions))


with open('rewards.txt', 'wt') as f:
    for x in res:
        # x[0]-up  x[1]-down  x[2]-disa  x[3]-terminal  x[4]-actions

        # greeting
        if x[0] == x[1] == x[2] == x[3] == 0:
            # 选对动作
            if x[4] == "greeting":
                f.write(str(x) + ", 80" + "\n")
            else:
                f.write(str(x) + ", -80" + "\n")

        # end
        elif x[3] == 1:
            if x[4] == "end":
                f.write(str(x) + ", 100" +"\n")
            else:
                f.write(str(x) + ", -100" + "\n")

        # slot full - end
        elif not x[0] and not x[1] and not x[2]:
            if x[4] == "end":
                f.write(str(x) + ", 100" + "\n")
            else:
                f.write(str(x) + ", -100" + "\n")

        # up, intent == 1 条件类
        # 终止为0
        elif not x[3] and x[0] != 3 and x[1] == x[2] == 3:
            if x[0] == 0:
                if x[4] == "up_q":
                    f.write(str(x) + ", 80" + "\n")
                else:
                    f.write(str(x) + ", -80" + "\n")
            elif x[0] == 1:
                if x[4] == "up_yes":
                    f.write(str(x) + ", 80" + "\n")
                else:
                    f.write(str(x) + ", -80" + "\n")
            elif x[0] == 2:
                if x[4] == "up_no":
                    f.write(str(x) + ", 80" + "\n")
                else:
                    f.write(str(x) + ", -80" + "\n")
            else:
                print("error")

        # down, inent == 0 含义类
        elif not x[3] and x[0] == 3 and x[1] != 3 and x[2] != 3:
            # disambiguation
            if x[2] == 1:
                if x[4] == "down_disa":
                    f.write(str(x) + ", 80" + "\n")
                else:
                    f.write(str(x) + ", -80" + "\n")
            # elif shows x[1] != 1 now 这个x[1] 要么用户 要么材料 不会有未填充的状态
            elif x[1] == 0:
                if x[4] == "down_disa":
                    f.write(str(x) + ", 80" + "\n")
                else:
                    f.write(str(x) + ", -80" + "\n")
            elif x[1] == 1:
                pass
            elif x[1] == 2:
                pass
            else:
                print("error")





        else:
            f.write(str(x)+"\n")

print(len(res))









