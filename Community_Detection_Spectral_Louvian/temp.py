import random

def color_list(n):
    colors= []
    for  i in range(n):
        color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        colors.append(color)
    return colors


print(color_list(100))