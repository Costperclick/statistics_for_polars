count = {
    1:4,
    2:1,
    3:1,
    4:1
}

x = [k for k,v in count.items() if v == max(count.values())]
print(x)