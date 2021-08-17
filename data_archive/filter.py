import pandas as pd 
from blimpy import Waterfall
# Using readlines()
file1 = open('fine_cadences.txt', 'r')
Lines = file1.readlines()


li = []
temp =0
for el in range(len(Lines)):
    if el%7==0:
        temp = el
        li.append(Lines[el].strip().replace('----- cadence: ','').replace("--- cadence: ",''))
    else:
        index = Lines[el].strip().find(':')
        li.append(Lines[el].strip()[index+2:].replace('*','').replace(' ',''))

total = {}
for i in range(len(li)//7):
    temp= li[i*7:(i+1)*7]
    total[temp[0]+"-"+str(i)] = temp[1:7]


final ={}
for key in total:
    fchan = Waterfall('../../../../../../..'+total[key][0], load_data= False).header['fch1']
    flag = True
    for i in range(6):
        time =  Waterfall('../../../../../../..'+total[key][i], load_data= False).selection_shape[0]
        if time!=16:
            flag = False
    if flag:
        final[key] = total[key]

# final ={}
# for key in total:
#     fchan = Waterfall('../../../../../../..'+total[key][0], load_data= False).header['fch1']
#     if fchan< 2000:
#         final[key] = total[key]
    


df = pd.DataFrame(final)
df.to_csv("fine_cadences_formatted_updated_big.csv", )