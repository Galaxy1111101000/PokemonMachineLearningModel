import os

for i in range (10, 13):
    for j in range (1,31):

        month = str(i)
        if j < 10:
            day = "0" + str(j)
        else:
            day = str(j)

        path = fr"C:\Users\aneya\PycharmProjects\pythonProject1\python-info-scraper\prices-2025-{month}-{day}.ppmd.7z"
        os.remove(path)