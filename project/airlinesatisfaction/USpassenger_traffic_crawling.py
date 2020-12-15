# https://www.tsa.gov/coronavirus/passenger-throughput?page=0

import csv 
import requests
from bs4 import BeautifulSoup as bs
import re

for i in range(1,-1,-1):
    page = requests.get(f'https://www.tsa.gov/coronavirus/passenger-throughput?page={i}')
    soup = bs(page.text, 'html.parser')

    d = re.compile("\d+/\d+")
    date = soup.findAll("td",{"class": "views-field views-field-field-today-date"})
    date_data = [d.match(element.text.strip()).group() for element in date]

    number = soup.findAll("td",{"views-field views-field-field-this-year"})
    num_data = [element.text.strip().replace(",","") for element in number]

    f = open('tsa_traffic_data.csv','w',newline='')
    if i==1:
        wr = csv.writer(f)
        wr.writerow(["Date","Number of passengers"])

    for j in range(len(date_data)-1,-1,-1):
        data=[date_data[j],num_data[j]]
        wr.writerow(data)

f.close()
