import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']


def process(m):
    # names = ['5', '10', '15', '20', '25']
    x = range(len(m))
    y = m

    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    plt.ylim(0,1)  # 限定纵轴的范围          here
    plt.plot(x, y, mec='r', mfc='w')

    # plt.scatter(x,y)

    # plt.legend()  # 让图例生效
    # plt.xticks(x, names, rotation=45)
    plt.margins(0)
    # plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"every 100 episode")  # X轴标签
    plt.ylabel("action precision")  # Y轴标签
    # plt.title("A simple plot")  # 标题

    plt.show()

# process([1,2,3,4,5])

