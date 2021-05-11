import requests
import csv, json
import pandas as pd
from numba import jit, prange, njit
import time
import os.path
from os import path

# Search function to parse the URL's
@jit(nopython=True)
def find_slash(num, string):
    count = 0
    index = []
    for char in string:
        if char =='/':
            index.append(count)
        count+=1
    return index[num]




# api call request to filter for L band and to get the fine resolution gbt data with complete cadences.
data = requests.get("http://seti.berkeley.edu/opendata/api/query-files",
    params ={'target':"","telescope":"GBT","cadence":True, 
            "file_type":"HDF5",
            "center_freq":1475.09765625,
            'primaryTarget':True,
            'grades':'fine'})
#convert to strings
s = data.text
json_acceptable_string = s.replace("'", "\"")
d = json.loads(json_acceptable_string)

#filters for cadence links
cadence_links = []
df = pd.DataFrame.from_dict(d["data"][1:])

# adds in the cadence urls which are API's
for i in range(len(df["cadence_url"])):
    cadence_links.append(df["cadence_url"][i])

# cadence is filtered for no repeates
cadence_links = list(set(cadence_links))
print("Total Number of Cadences: "+str(len(cadence_links)))

data = requests.get(cadence_links[i])
s = data.text
json_acceptable_string = s.replace("'", "\"")
d = json.loads(json_acceptable_string)
df = pd.DataFrame.from_dict(d["data"])


# cadence individual files are now checked and added into the list
cadence_directory = []
count = 0
for i in range(len(cadence_links)):
    if count%10==0:
        print(count)
    count+=1
    data = requests.get(cadence_links[i])
    s = data.text
    json_acceptable_string = s.replace("'", "\"")
    d = json.loads(json_acceptable_string)
    df = pd.DataFrame.from_dict(d["data"])

    # convert the url link into a usuable DIRECTORY within the data storage
    directory=[]
    for i in range(len(df['url'])):
        string = df['url'][i].replace("http://","").replace(".ssl.berkeley.edu","")
        index = find_slash(0,string)
        string = string.replace(string[0:index], "mnt_"+ string[0:index] )
        index = index+4
        index1 = find_slash(1,string)
        if string[index:index1] == "/dl2":
            string = string.replace(string[index:index1], "/datax2/dl" )
        elif  string[index:index1]  == "/dl":
            string = string.replace(string[index:index1], "/datax/dl" )
        else:
            print(string)
            print("funky")

        directory.append(string)
    cadence_directory.append([df['target'][0], directory])


# forms a dictionary to convert into the CSV
dic ={}
for i in range(len(cadence_directory)):
    if len(cadence_directory[i][1]) == 6: 
        dic[cadence_directory[i][0]] = cadence_directory[i][1]

# save temporary in file
compact =pd.DataFrame.from_dict(dic) 
compact.to_csv("../L_band_directory.csv")

# read the file
df = pd.read_csv('../L_band_directory.csv')
headers_list = df.columns[2:]

#checks if files are ACTUALLY present
print(headers_list)
def check_director(list_dir):
    bad =[]
    for directory in list_dir:
        if path.exists("../../../../../../../" + directory):
            continue
        else:
            bad.append("../../../../../../../" + directory)
    if len(bad)==0:
        return True
    else:
        return False

for header in headers_list:
    if check_director(df[header]):
        continue
    else:
        print("removing invalid Directory "+ header)
        df.pop(header)

# save new file and replace old one
df.to_csv("../L_band_directory.csv")