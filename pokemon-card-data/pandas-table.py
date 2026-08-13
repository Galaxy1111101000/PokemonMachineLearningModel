import datetime
import json
import pandas as pd

dfs = []
number_giratina = 30
number_cards = 13178
avg_cards = 181
bw = 812

def getDate(date):
    d1 = datetime.datetime(year=2025, month=1, day=1)
    return (date-d1).total_seconds()/3600

for i in range(1, 13):
    for j in range(1, 30):
        if i == 1 and j == 24:
            continue
        if (i == 1 or i == 2 or i == 5 or i == 6)and j == 28:
            break
        if i == 7 and j == 25:
            break

        if i < 10:
            month = "0" + str(i)
        else:
            month = str(i)
        if j < 10:
            day = "0" + str(j)
        else:
            day = str(j)

        with open (rf'C:\Users\aneya\PycharmProjects\pythonProject1\python-info-scraper\2025-{month}-{day}\3\3118\prices', 'r') as json_file:
            data = json.load(json_file)

        giratina_data = ''

        for item in data["results"]:
            if item['productId'] == 284137:
                print(item)
                giratina_data = item
                giratina_data["date"] = getDate(datetime.datetime(year=2025, month=i, day=j))

                if (i == 1 and j == 17) or (i == 3 and j == 28) or (i == 5 and j == 30) or (i == 9 and j == 26) or (i == 11 and j == 14):
                    number_cards += avg_cards
                elif (i == 7 and j == 18):
                    number_cards += bw

                giratina_data["percent"] = round(float(number_giratina/number_cards),9)
                temp = pd.DataFrame([giratina_data])
                dfs.append(temp)
                continue

data_final = pd.concat(dfs, ignore_index=True)
data_final.dropna()
print(data_final)
data_final.to_csv("historic-data.csv", mode="a", index=True)