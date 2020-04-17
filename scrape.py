# imports
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import pandas as pd
import numpy as np

# get list of schools
schools = set()

response = requests.get("https://www.lawschooldata.org/school")
soup = BeautifulSoup(response.text, "html.parser")
for option in soup.find_all('option'):
    schools.add(option['value'].split('=')[2])

schools = list(schools)

with open("file_home.html", "w") as file:
    file.write(str(soup))

import numpy as np
#username, decision, gpa, lsat, urm, work_experience, scholarship money, sent, recieved, completed, decision
rows = []

for cycle_id in range(12,18): #12 is 2014 to 2015
    for school_iter in range(len(schools)):
        url_for_school = f'https://www.lawschooldata.org/school/applicants?cycle_id={cycle_id}&school={schools[school_iter]}'
        response_school = requests.get(url_for_school)
        soup_school = BeautifulSoup(response_school.text, "html.parser")
        raw_table = str(soup_school.findAll(lambda tag: tag.name=='script')[19]).split('data')[1][8:].split('order')[0].split('href')[1:-1]
        for j in range(len(raw_table)):
            row = []
            for i,elem in enumerate(raw_table[j].split('\n')):
                if i == 0:
                    row.append(elem.split('\n')[0].split('"')[1].split('/')[-1])
                if i in [1]:
                    row.append(elem.replace("'","").replace(',','').lstrip().rstrip())
                if i in [2,3]:
                    try:
                        row.append(float(elem.split('\'')[1]))
                    except ValueError:
                        row.append(np.nan)
                    gpa_or_lsat = elem.split('\'')[1]
                # under rep minority
                if i == 4:
                    if "no" in elem:
                        row.append(0)
                    else:
                        row.append(1)
                if i ==5:
                    row.append(elem.replace("'","").replace(',','').lstrip().rstrip())
                if i ==6:
                    row.append(elem.split()[0].replace("'","").replace(',','').replace('$','').lstrip().rstrip())
                if i in [7,8,9]:
                    row.append(elem.split('\'')[1])
                if i == 12:
                    row.append(elem.split('\'')[1])
            row.append(' '.join(schools[school_iter].split('+')))
            row.append(cycle_id)

            rows.append(row)

df_full = pd.DataFrame(data = rows, columns = ['username', 'decision', 'gpa', 'lsat', 'urm', 'work_experience', 'scholarship_money', 'sent', 'recieved', 'completed', 'decision', 'school', 'cycleid'])

df_full.to_csv('data_full.csv')