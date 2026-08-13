import os
import shutil

for i in range(9,13):
    if i / 10.0 < 1:
        month_str = "0" + str(i)
    else:
        month_str = str(i)

    for j in range(1, 31):
        if i == 2:
            if j == 28:
                break
        if j / 10.0 < 1:
            day_str = "0" + str(j)
        else:
            day_str = str(j)
        working_dir = f"C:\\Users\\aneya\PycharmProjects\pythonProject1\python-info-scraper\\2025-{month_str}-{day_str}"
        retain = ["3", "23353", "3118", "23228"]


        other_dir = r"C:\Users\aneya\PycharmProjects\pythonProject1\python-info-scraper\prices-2024-02-08.ppmd.7z"

        os.chdir(working_dir)

        ## isolates 3

        for item in os.listdir(os.getcwd()):
            if item not in retain:
                shutil.rmtree(item)

        ## isolates only the sets I want
        os.chdir(working_dir + r"\3")
        for item in os.listdir(os.getcwd()):
            if item not in retain:
                shutil.rmtree(item)
