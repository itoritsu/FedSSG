n=1
v = 200000

for n in range (1,10000000):
    v = v * 0.9
    if v <=1:
        break
    else:
        n+=1
print(n)