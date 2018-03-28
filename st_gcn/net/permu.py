import numpy as np
num_node = 25
self_link = [i for i in range(num_node)]
perm = [[[], []] for i in range(num_node-1)]

for dilation in range(1, 25):
    for i in range(num_node+1):
        index = (i * dilation)%(num_node)
        node = self_link[index]
        perm[dilation - 1][0].append(node)
    xx = perm[dilation - 1][0][0:num_node]
    xxx = sorted(xx)
    print ("here")
print (self_link)