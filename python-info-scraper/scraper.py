import subprocess
import os
import py7zr
import shutil

day = 8
month = 2
year = 2024


for i in range(8,13):
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

        url = f"https://tcgcsv.com/archive/tcgplayer/prices-2025-{month_str}-{day_str}.ppmd.7z"
        cmd = ["curl", "-O", url]
        result = subprocess.run(cmd, capture_output=True, text=True)

        output_dir = r"C:\Users\aneya\PycharmProjects\pythonProject1\python-info-scraper"
        filename = os.path.basename(url)

        with py7zr.SevenZipFile(filename, mode='r') as archive:
            archive.extractall(path=output_dir)




