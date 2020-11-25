'''
You should download the weekly dataset from the following site:
ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_weekly_mlo.txt
and extract the data of the last two years to create a file named co2w.txt.

Because the above link is blocked, I personally use a data "Globally averaged marine surface
monthly mean data" from Global Monitoring Laboratory website.
https://www.esrl.noaa.gov/gmd/ccgg/trends/gl_data.html

ASSIGNMENT:
You should plot a graph with x-axis(year_month_day) and y-axis(co2) using co2w.txt
'''
import matplotlib.pyplot as plt
import pandas as pd

# extract the data of the last two years to create a file named co2w.txt
with open("co2_mm_gl.txt","r",encoding="utf-8") as co2:
    data=[]
    for i in co2:
        a,b,c,d,e=i.split()
        if (a!="year" and int(b)>11 and int(a)==2017) or (a!="year" and int(a)>=2018):
            data.append(a+"_"+b+","+e)

with open("co2w.txt","w",encoding="utf-8") as f:
    f.write("\n".join(data))

# plot a graph
data=pd.read_csv("co2w.txt")
data.columns=["year_month","co2"]
plt.title("Globally averaged CO2 records of the last 2 years")
plt.grid()
plt.xlabel("year_month")
plt.ylabel("density")
plt.xticks(rotation=45,fontsize=10)
plt.plot(data["year_month"],data["co2"],"ro")
fig=plt.figure(1)
fig.set_size_inches(15,10)
plt.savefig("co2w.png",dpi=fig.dpi,bbox_inches="tight")
# plt.show()