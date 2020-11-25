# print co2 all data only

with open("co2_v2.txt","r",encoding="utf-8") as f:
    data=f.readlines()
    for i in data:
        print(i.partition(",")[2].strip())
