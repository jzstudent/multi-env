import matplotlib.pyplot as plt
import math
import argparse
def shape_centroid(i):
    plt.figure(figsize=(8, 8))
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.title("Please decide agents' location")
    pos=plt.ginput(i)
    position=[]
    for a in pos:
        position.append(list(a))
    x=[]
    y=[]
    for i in range(len(position)+1):
        if i==len(position):
            i=0
        x.append(position[i][0])
        y.append(position[i][1])
    plt.close()
    plt.figure(figsize=(8, 8))
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title("Please decide object's centroid")
    plt.plot(x,y,color="r")
    plt.scatter(x, y, color="b")
    centroid=plt.ginput(1)
    centroid=list(centroid[0])
    plt.close()
    del x[-1]
    del y[-1]
    return x,y,centroid
parser = argparse.ArgumentParser()
parser.add_argument("num_agents", type=int)
config = parser.parse_args()
x,y,c=shape_centroid(config.num_agents)
with open("shape.txt", "w") as f:
    x = str(x)
    y = str(y)
    c = str(c)
    f.writelines("{}\n".format(x))
    f.writelines("{}\n".format(y))
    f.writelines("{}\n".format(c))
