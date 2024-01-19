list = [5, 3 , 1]
result = []
for layer in range(1, len(list)):
    print(layer)
    res = list[layer - 1]
    print(res)
    result.append(res)
print(result)
