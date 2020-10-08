# subtract each element (string) in two lists

a=["doyoon","lee"]
b=["south","korea"]
for i in range(len(a)):
    for j in b[i]:
        if j not in a[i]:
            continue
        else:
            print(f"delete {j}")
            a[i]=a[i].replace(j,"")
print(a)
