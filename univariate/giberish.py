import random


n = [4, 2, 6, 9, 10, 21]

for i in range(100):
    rand= random.choices(n, k=len(n))
    print(rand)
