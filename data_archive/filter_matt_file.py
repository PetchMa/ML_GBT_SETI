import pandas as pd 
# Using readlines()
file1 = open('fine_cadences.txt', 'r')
Lines = file1.readlines()


li = []
temp =0
for el in range(len(Lines)):
    if el%7==0:
        temp = el
        li.append(Lines[el].strip().replace('----- cadence: ',''))
    else:
        index = Lines[el].strip().find(':')
        li.append(Lines[el].strip()[index+2:].replace('*',''))

total = {}
for i in range(len(li)//7):
    temp= li[i*7:(i+1)*7]
    total[temp[0]] = temp[1:7]


df = pd.DataFrame(total)

df.to_csv('fine_cadences_formated.csv')


# li = []
# temp =0
# for el in range(len(Lines)):
#     if el%7==0:
#         temp = el
#         li.append(Lines[el].strip().replace('----- cadence: ',''))
#     else:
#         index = Lines[el].strip().find(':')
#         li.append(Lines[el].strip()[index+2:].replace('*',''))

# print(li)

