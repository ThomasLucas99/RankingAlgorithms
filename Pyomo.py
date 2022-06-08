#pyomo
#from __future__ import division
import pyomo.environ as pyo
import time
import sys
import pprint
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from pyomo.common.tempfiles import TempfileManager
TempfileManager.tempdir = "D:/Memoire"
from PythonFunctions import *

def ReadData(file):
    f = open(file, "r", encoding="utf-8")
    line = f.readline()
    line = f.readline()
    line = line.split(",")
    read = [line]
    
    while f:
        line = f.readline()
        line = line.split(",")
        if line == [""]:
            break
        read += [line]
    f.close()
    return read


# Can only be after 1900
def ReadDataFromDate(file, date):
    f = open(file, "r", encoding="utf-8")
    line = f.readline()
    i = 0
    while f:
        line = f.readline()
        line = line.split(",")
        if i<129:
            i += 1
        elif dt.datetime.strptime(line[0], '%d/%m/%Y') > date:
            break
    read = [line]
    
    while f:
        line = f.readline()
        line = line.split(",")
        if line == [""]:
            break
        read += [line]
    f.close()
    return read


def ReadDataDates(file, start, end):
    f = open(file, "r", encoding="utf-8")
    line = f.readline()
    i = 0
    while f:
        line = f.readline()
        line = line.split(",")
        if i<129:
            i += 1
        elif dt.datetime.strptime(line[0], '%d/%m/%Y') > start:
            break
    read = [line]
    
    while f:
        line = f.readline()
        line = line.split(",")
        if line == [""] or dt.datetime.strptime(line[0], '%d/%m/%Y') > end:
            break
        read += [line]
    f.close()
    return read


def ReadDataFromWCQ(file, date):
    f = open(file, "r", encoding="utf-8")
    line = f.readline()
    i = 0
    while f:
        line = f.readline()
        line = line.split(",")
        if i<129:
            i += 1
        elif "FIFA World Cup" in line[5] and dt.datetime.strptime(line[0], '%d/%m/%Y') > date:
            break
    read = [line]
    
    while f:
        line = f.readline()
        line = line.split(",")
        if line == [""]:
            break
        if "FIFA World Cup" in line[5]:
            read += [line]
    f.close()
    return read


def ReadDataFromWC(file, date):
    f = open(file, "r", encoding="utf-8")
    line = f.readline()
    i = 0
    while f:
        line = f.readline()
        line = line.split(",")
        if i<129:
            i += 1
        elif "FIFA World Cup" == line[5] and dt.datetime.strptime(line[0], '%d/%m/%Y') > date:
            break
    read = [line]
    
    while f:
        line = f.readline()
        line = line.split(",")
        if line == [""]:
            break
        if "FIFA World Cup" == line[5]:
            read += [line]
    f.close()
    return read

def ReadDataDatesWC(file, start, end):
    f = open(file, "r", encoding="utf-8")
    line = f.readline()
    i = 0
    while f:
        line = f.readline()
        line = line.split(",")
        if i<129:
            i += 1
        elif "FIFA World Cup" == line[5] and dt.datetime.strptime(line[0], '%d/%m/%Y') > start:
            break
    read = [line]
    
    while f:
        line = f.readline()
        line = line.split(",")
        if line == [""] or dt.datetime.strptime(line[0], '%d/%m/%Y') > end:
            break
        if "FIFA World Cup" == line[5]:
            read += [line]

    f.close()
    return read

def MakeDataGreat(data, unique):
    new_data = []
    for i in range(len(data)):
        if data[i][2] > data[i][3]:
            new_data.append([data[i][1], data[i][0], int(data[i][2]) - int(data[i][3])])
        if data[i][2] < data[i][3]:
            new_data.append([data[i][0], data[i][1], int(data[i][3]) - int(data[i][2])])
        #if data[i][2] == data[i][3]:
        #    new_data.append([data[i][1], data[i][0], 0])

    for i in range(len(unique)):
        new_data.append([unique[i], unique[i], 1])
    return new_data

def ReadPremier(file):
    f = open(file, "r", encoding="utf-8")
    line = f.readline()
    line = f.readline()
    line = line.split(",")
    
    read = [line]

    while f:
        line = f.readline()
        line = line.split(",")
        if line == [""]:
            break
        read += [line]

    f.close()
    return read

def ReadResults(file, unique, N):
    rank = [""]*N
    f = open(file, "r")
    while f:
        line = f.readline()
        
        line = line.replace(" ", "")
        line = line.replace("\n", "")

        if line == "Variables:" or line == "":
            line = f.readline()
            line = f.readline()
            line = f.readline()
            break
    
    while f:
        line = f.readline()
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        if line == "Objectives" or line == "":
            break
        line = line.split(":")
        
        line2 = int(line[2].split(".")[0])
        if line2 == 1:
            
            line = line[0].split(",")
            line0 = line[0].split("(")[1]
            line1 = line[1].split(")")[0]
            rank[int(line1)] = unique[int(line0)]
        
    f.close()
    return rank
        

def Countries(data):
    read = np.array(data)
    countries = read[:, [1,2]]
    return np.unique(countries)


def MatchTable(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(d):
        if data[i][2] > data[i][3]:
            matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
        #if data[i][2] == data[i][3]:
        #    matrix[0][1] += 1
        if data[i][2] < data[i][3]:
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 1
    return matrix

def MatchTableImportance(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(d):
        if data[i][2] > data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 10
                # Importance during tournament final *20 maybe etc
            elif data[i][4] == "FIFA World Cup qualification":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 5
            elif data[i][4] == "Friendly":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
            else:
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 2
        if data[i][2] < data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 10
            elif data[i][4] == "FIFA World Cup qualification":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 5
            elif data[i][4] == "Friendly":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 1
            else:
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 2
    return matrix


def MatchTableSmall(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(d):
        if data[i][2] > data[i][3]:
            matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1/d
        if data[i][2] < data[i][3]:
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 1/d
    return matrix

def WR(w, l, matrix, matches, imp):
    matrix[w][l] = (matrix[w][l]* matches[w][l] +imp)/ (matches[w][l] +imp)
    matches[w][l] += imp
    matches[l][w] += imp
    return matrix, matches

def MatchTableWR(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    matches = [[0]*N for i in range(N)]
    for i in range(len(data)):
        if data[i][2] > data[i][3]:
            matrix, matches = WR(np.where(unique == data[i][0])[0][0], np.where(unique == data[i][1])[0][0], matrix, matches, 1)
        if data[i][2] < data[i][3]:
            matrix, matches = WR(np.where(unique == data[i][1])[0][0], np.where(unique == data[i][0])[0][0], matrix, matches, 1)
    return matrix

def MatchTableWRI(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    matches = [[0]*N for i in range(N)]
    for i in range(len(data)):
        if data[i][2] > data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix, matches = WR(np.where(unique == data[i][0])[0][0], np.where(unique == data[i][1])[0][0], matrix, matches, 10)
            elif data[i][4] == "FIFA World Cup qualification":
                matrix, matches = WR(np.where(unique == data[i][0])[0][0], np.where(unique == data[i][1])[0][0], matrix, matches, 5)
            elif data[i][4] == "Friendly":
                matrix, matches = WR(np.where(unique == data[i][0])[0][0], np.where(unique == data[i][1])[0][0], matrix, matches, 1)
            else:
                matrix, matches = WR(np.where(unique == data[i][0])[0][0], np.where(unique == data[i][1])[0][0], matrix, matches, 2)
        if data[i][2] < data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix, matches = WR(np.where(unique == data[i][1])[0][0], np.where(unique == data[i][0])[0][0], matrix, matches, 10)
            elif data[i][4] == "FIFA World Cup qualification":
                matrix, matches = WR(np.where(unique == data[i][1])[0][0], np.where(unique == data[i][0])[0][0], matrix, matches, 5)
            elif data[i][4] == "Friendly":
                matrix, matches = WR(np.where(unique == data[i][1])[0][0], np.where(unique == data[i][0])[0][0], matrix, matches, 1)
            else:
                matrix, matches = WR(np.where(unique == data[i][1])[0][0], np.where(unique == data[i][0])[0][0], matrix, matches, 2)
    return matrix

def MatchTableImportanceSmall(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(d):
        if data[i][2] > data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 10/d
            elif data[i][4] == "FIFA World Cup qualification":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 5/d
            elif data[i][4] == "Friendly":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1/d
            else:
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 2/d
        if data[i][2] < data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 10/d
            elif data[i][4] == "FIFA World Cup qualification":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 5/d
            elif data[i][4] == "Friendly":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 1/d
            else:
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 2/d
    return matrix

def MatrixNormalizationLine(matrix, N):
    for i in range(N):
        s = np.sum(matrix[i])
        if s != 0:
            matrix[i] = matrix[i]/s
    return matrix

def MatrixNormalizationCol(matrix, N):
    for i in range(N):
        s = np.sum(matrix[:,i])
        if s != 0:
            matrix[:,i] = matrix[:,i]/s
    return matrix

def MatrixNormalization(matrix, N):
    for i in range(100):
        matrix = MatrixNormalizationCol(matrix, N)
        matrix = MatrixNormalizationLine(matrix, N)
    return matrix

def RemoveNoWinDraws(matrix, draws, m, unique, N):
    countries = []
    last = []
    while True:
        countries = NoWinDraws(matrix, draws, unique, N)
        if countries is None:
            break
        else:
            last.append(countries)
            N -= len(countries)
            i = 0
            while i < len(countries):
                index = np.where(unique == countries[i])[0][0]
                matrix = np.delete(matrix, index, axis = 0)
                matrix = np.delete(matrix, index, axis = 1)
                m = np.delete(m, index, axis = 0)
                m = np.delete(m, index, axis = 1)
                unique = np.delete(unique, index)
                i += 1
    return matrix, m, unique, N, last

def NoWin(matrix, unique, N):
    i = 0
    countries = []
    for i in range(len(unique)):
        if np.sum(matrix[i]) == 0:
            countries.append(unique[i])
    
    if len(countries) == 0:
        return None
    return countries

def NoWinDraws(matrix, draws, unique, N):
    i = 0
    countries = []
    for i in range(len(unique)):
        if np.sum(matrix[i]) == 0 and np.sum(draws[i]) == 0 and np.sum(draws[:,i]) == 0:
            countries.append(unique[i])
    
    if len(countries) == 0:
        return None
    return countries

def RemoveNoWin(matrix, m, unique, N):
    countries = []
    last = []
    while True:
        countries = NoWin(matrix, unique, N)
        if countries is None:
            break
        else:
            last.append(countries)
            N -= len(countries)
            i = 0
            while i < len(countries):
                index = np.where(unique == countries[i])[0][0]
                matrix = np.delete(matrix, index, axis = 0)
                matrix = np.delete(matrix, index, axis = 1)
                m = np.delete(m, index, axis = 0)
                m = np.delete(m, index, axis = 1)
                unique = np.delete(unique, index)
                i += 1
    return matrix, m, unique, N, last

def RemoveGames(data, countries, d):
    i = 0
    removed = 0
    while i < d:
        if data[i][0] in countries or data[i][1] in countries:
            data = np.delete(data, i, axis = 0)
            d -= 1
            i -= 1
            removed +=1
        i += 1

    return data, d, removed
    
def Flat(l):
    flat = []
    for i in range(len(l)):
        for j in range(len(l[i])):
            flat.append(l[i][j])
    return flat

def WriteCSV(file, matrix, unique, N):
    f = open(file, "w+", encoding="utf-8")
    f.write("Teams,")
    for i in range(N):
        f.write(str(unique[i]) + ",")

    for i in range(N):
        f.write("\n" + str(unique[i]) + ",")
        for j in range(N):
            f.write(str(matrix[i][j]) + ",")

    f.close()

def StimesU(u, s, n, N):
    if n > len(s):
        return []
    vec = np.zeros(N)
    for i in range(n):
        vec += s[i] * np.absolute(u[:,i])
    return vec
    

def sumlc(matrix, l, c):
    s = np.sum(matrix[l])
    s += np.sum(matrix[:, c])
    return s

def ExtractRankingFile(file, unique, m, matrix, date, data, d):
    f = open(file, "r")
    line = f.readline()
    line = f.readline()
    line = line.split(",")
    read = [line]
    date = date - relativedelta(years = 2)
    while f:
        line = f.readline()
        line = line.split(",")
        line[8] = line[8][:-1]
        if dt.datetime.strptime(line[8], '%d/%m/%Y') > date:
            break
    read = [line]
    while f:
        line = f.readline()
        line = line.split(",")
        if line == [""]:
            break
        read += [line]
    f.close()
    read = np.array(read)
    read = read[:, 1:3]
    N = len(read)
    i = 0

    while i < N:
        if read[i,1] not in unique:
            read = np.delete(read, i, 0)
            i -= 1
            N -= 1
        i += 1

    i = 0
    u = len(unique)
    removed_teams = []
    while i < u:
        if unique[i] not in read[:, 1]:
            removed_teams.append(unique[i])
            unique = np.delete(unique, i, 0)
            matrix = np.delete(matrix, i, 0)
            matrix = np.delete(matrix, i, 1)
            m = np.delete(m, i, 0)
            m = np.delete(m, i, 1)
            u -= 1
            i -= 1
        i += 1
    data, d, removed = RemoveGames(data, removed_teams, d)
    i = 0
    rank = np.array([["0", "0"] for i in range(u)], dtype=object)
    while i < u:
        last = np.where(read[:, 1] == unique[i])[0][-1]
        if last == "":
            print("STOP")
        rank[i] = [read[int(last)][0], read[int(last)][1]]
        i+=1
    rank = np.array(rank)
    vec = rank[:,0].astype(np.intc)
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    rank = [tple[0] for tple in ranked]
    return rank, unique, m, matrix, u, data, d

#Change upsets sending only an array of int
def Upsetspyo(matrix, r, unique, N, d):
    upsets = 0
    for i in range(N):
        print(i)
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    if k>l:
                        upsets += (r[i,k] * r[j,l]) * matrix[i][j]
                        
    print("Upsets")
    print(upsets)
    return upsets

def Score(matrix, r, unique, N, d):
    score = 0
    for i in range(N):
        print(i)
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    score += (r[i,k] * r[j,l]) * matrix[i][j] * (k - l)
                        
    print("Score computed")
    #print(upsets)
    return score

def Minimum(matrix, unique):
    minim = 0
    for i in range(len(unique)):
        j = i + 1
        while j < len(unique):
            if matrix[i][j] > 0 and matrix[j][i] > 0:
                minim += min(matrix[j][i], matrix[i][j])
            j += 1
    return minim
  
def Merge(cont1, cont2, unique, matrix):
    print("merge")
    cont1 = np.array(cont1)
    cont2 = np.array(cont2)
    merged = cont1.copy()
    merged = merged.tolist()
    N = len(cont1) + len(cont2)
    score = np.zeros(len(cont1))
    place = 0
    for i in range(len(cont2)):
        insert = np.where(unique == cont2[i])[0][0]
        """
        for j in range(len(cont1)):
            
            a = np.where(unique == cont1[j])[0][0]
            score[0] += matrix[a][insert]
            score[0] -= matrix[insert][a]
        """
        j = 0
        score = np.zeros(len(cont1))
        while j < len(cont1):
            """
            print(unique)
            print(cont1)
            print(cont1[j])
            print(j)
            """
            a = np.where(unique == cont1[j])[0][0]
            """
            print("truc")
            print(a)
            print(insert)
            print(j)
            print(len(cont1))
            print(len(matrix))
            print(len(matrix[insert]))
            print(len(score))
            """
            score[j] = score[j-1] + matrix[insert][a] - matrix[a][insert]
            j += 1
        new_score = score[place:]
        m = min(new_score)
        places = np.where(new_score == m)[0]
        for k in range(len(places)):
            place = places[k] + place
            insplace = place + i
            merged.insert(insplace, cont2[i])
            break
    return merged

file = "Data/FIFA/Foot.csv"
#file = "Data/Premier/2017_2018.csv"
#start = dt.datetime(1900, 1, 1)
date = dt.datetime(2018, 1, 1)
#end = dt.datetime(2021, 1, 1)

read = np.array(ReadDataFromDate(file, date))
#read = np.array(ReadDataFromWC(file, date))
#read = np.array(ReadDataFromWCQ(file, date))
#read = np.array(ReadDataDates(file, date, end))
#read = np.array(ReadDataDatesWC(file, date, end))

#read = np.array(ReadPremier(file))

#read = read[60:]
#read = read[:45]

#64 matches and 32 teams = 340 seconds
#68 matches, 33 teams = 2511 seconds

real_rank = ["Belgium", "Brazil", "France", "England", "Argentina", "Italy", "Spain", "Portugal", "Denmark", "Netherlands",
"United States", "Germany", "Switzerland", "Mexico", "Croatia", "Colombia", "Uruguay", "Sweden", "Wales", "Senegal",
"Iran", "Peru", "Serbia", "Chile", "Ukraine", "Japan", "Poland", "Morocco", "Algeria", "Tunisia",
"Austria", "Czech Republic", "South Korea", "Russia", "Australia", "Nigeria", "Turkey", "Scotland", "Hungary", "Canada",
"Norway", "Slovakia", "Paraguay", "Romania", "Egypt", "Ecuador", "Ireland", "Qatar", "Costa Rica", "Cameroon",
"Saudi Arabia", "Ghana", "Mali", "Nothern Ireland", "Greece", "Ivory Coast", "Jamaica", "Finland", "Venezuela", "Burkina Faso",
"Bosnia and Herzegovina", "Iceland", "Panama", "DR Congo", "Slovenia", "Albania", "North Macedonia", "South Africa", "UAE", "El Salvador",
"Bulgaria", "Montenegro", "Cape Verde", "China PR", "Iraq", "Honduras", "Bolivia", "Israel", "Oman", "Curacao",
"Guinea", "Uganda", "Benin", "Uzbekistan", "Georgia", "Syria", "Haiti", "Zambia", "Gabon", "Jordan",
"Bahrain", "Armenia", "Luxembourg", "Belarus", "Lebanon", "Kyrgyzstan", "Congo", "Vietnam", "Palestine", "Trinidad and Tobago",
"Madagascar", "Kenya", "Mauritania", "India", "Cyprus", "Guinea-Bissau", "Estonia", "Sierra Leone", "North Korea", "New Zealand",
"Kosovo", "Namibia", "Niger", "Equa Guinea", "Thailand", "Tajikistan", "Libya", "Mozambique", "Kazakhstan", "Azerbaijan",
"Zimbabwe", "Guatemala", "Faroe Islands", "Togo", "Sudan", "Angola", "Antigua and Barbuda", "Philippines", "Malawi", "CAR",
"Tanzania", "Comoros", "Turkmenistan", "Latvia", "Rwanda", "Lithuania", "Ethiopia", "Saint Kitts and Nevis", "Suriname", "Burundi",
"Solomon Islands", "Kuwait", "Nicaragua", "Liberia", "Lesotho", "Eswatini", "Hong Kong", "Botswana", "Afghanistan", "Gambia",
"Yemen", "Myanmar", "New Caledonia", "Malaysia", "Andorra", "Dominican Republic", "Maldives", "Chinese Taipei", "Tahiti", "Singapore",
"Fiji", "Barbados", "Vanuatu", "Indonesia", "Papua New Guinea", "Bermuda", "South Sudan", "Grenada", "Nepal", "Belize",
"Cambodia", "Mauritius", "Puerto Rico", "Saint Vincent", "Malta", "Guyana", "Saint Lucia", "Montserrat", "Cuba", "Chad",
"Moldova", "Macau", "Dominica", "Mongolia", "Bhutan", "Bangladesh", "Laos", "Brunei", "Sao Tome", "American Samoa",
"Liechtenstein", "Djibouti", "Samoa", "Somalia", "Cayman Islands", "Timor-Leste", "Seychelles", "Pakistan", "Tonga", "Aruba",
"Bahamas", "Eritrea", "Gibraltar", "Sri Lanka", "Turks and Caicos", "Guam", "US Virgin Islands", "British Virgin Islands", "San Marino", "Anguilla"]


UEFA1 = ["Belgium", "France", "England", "Italy", "Spain", "Portugal", "Denmark", "Netherlands", "Germany", "Switzerland",
"Croatia", "Sweden", "Wales"]
UEFA2 = ["Serbia", "Ukraine", "Poland", "Austria", "Czech Republic", "Russia", "Turkey", "Scotland", "Hungary", "Norway",
"Slovakia", "Romania", "Ireland", "Nothern Ireland"]
UEFA3 = ["Greece", "Finland", "Bosnia and Herzegovina", "Iceland", "Slovenia", "Albania", "North Macedonia", "Bulgaria", "Montenegro","Israel",
"Georgia", "Armenia", "Luxembourg", "Belarus"]
UEFA4 = ["Cyprus", "Estonia", "Kosovo", "Kazakhstan", "Azerbaijan", "Faroe Islands", "Latvia", "Lithuania", "Andorra", "Malta",
"Moldova", "Liechtenstein", "Gibraltar", "San Marino"]

CONMEBOL = ["Brazil", "Argentina", "Colombia", "Uruguay",  "Peru",  "Chile", "Paraguay", "Ecuador", "Venezuela", "Bolivia"]

CONCACAF1 = ["United States", "Mexico", "Canada", "Costa Rica", "Jamaica", "Panama", "El Salvador", "Honduras", "Curacao", "Haiti", "Trinidad and Tobago",
"Guatemala"]
CONCACAF2 = ["Antigua and Barbuda", "Saint Kitts and Nevis", "Suriname", "Nicaragua", "Dominican Republic", "Barbados", "Bermuda", "Grenada", "Belize", "Puerto Rico",
"Saint Vincent"]
CONCACAF3 = ["Guyana", "Saint Lucia", "Montserrat", "Cuba", "Dominica", "Cayman Islands", "Aruba", "Bahamas", "Turks and Caicos", "US Virgin Islands", "British Virgin Islands",
"Anguilla"]

CAF1 = ["Senegal",  "Morocco", "Algeria", "Tunisia", "Nigeria", "Egypt", "Cameroon", "Ghana", "Mali", "Ivory Coast",
"Burkina Faso", "DR Congo",  "South Africa"]
CAF2 = ["Cape Verde", "Guinea", "Uganda", "Benin", "Zambia", "Gabon", "Congo", "Madagascar", "Kenya", "Mauritania",
"Guinea-Bissau", "Sierra Leone", "Namibia", "Niger"]
CAF3 = ["Equa Guinea", "Libya", "Mozambique", "Zimbabwe", "Togo", "Sudan", "Angola", "Malawi", "CAR", "Tanzania",
"Comoros", "Rwanda", "Ethiopia", "Burundi"]
CAF4 = ["Liberia", "Lesotho", "Eswatini", "Botswana", "Gambia", "South Sudan", "Mauritius", "Chad", "Sao Tome", "Djibouti",
"Somalia", "Seychelles", "Eritrea"]

AFC1 = ["Iran", "Japan", "South Korea", "Australia", "Qatar", "Saudi Arabia", "UAE", "China PR", "Iraq", "Uzbekistan",
"Oman", "Syria", "Jordan", "Bahrain", "Lebanon"]
AFC2 = ["Kyrgyzstan", "Vietnam", "Palestine", "India", "North Korea", "Thailand", "Tajikistan", "Philippines", "Turkmenistan", "Kuwait",
"Hong Kong", "Afghanistan", "Yemen", "Myanmar", "Malaysia"]
AFC3 = ["Maldives", "Chinese Taipei", "Singapore", "Indonesia", "Nepal", "Cambodia", "Macau", "Mongolia", "Bhutan", "Bangladesh",
"Laos", "Brunei", "Timor-Leste", "Pakistan", "Sri Lanka", "Guam"]

OFC = ["New Zealand", "Solomon Islands", "New Caledonia", "Tahiti", "Fiji", "Vanuatu", "Papua New Guinea", "American Samoa", "Samoa", "Tonga"]

real = [OFC, AFC3, AFC2, AFC1, CAF4, CAF3, CAF2, CAF1, CONCACAF3, CONCACAF2, CONCACAF1, UEFA4, CONMEBOL, UEFA3, UEFA2, UEFA1]
#read = read[0:130]
t = 100
i = 0
#unique = np.array(real_rank)
data = read[:, 1:6]
print(data)

data = [
    ["a", "b", "1", "0", "cup"],
    ["a", "d", "1", "0", "cup"],
    ["b", "c", "1", "0", "cup"],
    ["b", "d", "1", "0", "cup"],
    ["c", "a", "1", "0", "cup"],
    ["d", "c", "1", "0", "cup"]
]

d = len(data)
#data, d = KeepOnlyGames(data, real_rank, d)

new_unique = []
#unique = np.array(Countries(data))
#print(unique)
unique = ["a", "b", "c", "d"]
print(d)

m = np.array([[0,1,1,0],
 [0,0,1,1],
 [1,0,0,0],
 [0,0,1,0]])

matrix = m.copy()
#matrix = np.array(MatchTable(data, unique, len(unique)))
#m = np.array(MatchTable(data, unique, len(unique)))
UN = len(unique)
N = len(unique)
"""
r, unique, m, matrix, N, data, d = ExtractRankingFile("fifa_ranking.csv", unique, m, matrix, date, data, d)

#test = unique
matrix, m, unique, N, last = RemoveNoWin(matrix, m, unique, N)
#matrix, m, unique, N, first = RemoveOnlyWin(matrix, m, unique, N)
fl = Flat(last)
#flat_first = Flat(first)
data, d, removed_games = RemoveGames(data, fl, d)
#data, d, removed_games = RemoveGames(data, flat_first, d)
"""
csv = "Matrix.csv"

WriteCSV(csv, matrix, unique, N)

print(d)
print(N)
model = pyo.ConcreteModel()

model.I = pyo.RangeSet(0, N-1)
model.J = pyo.RangeSet(0, N-1)

model.r = pyo.Var(model.I, model.J, domain = pyo.NonNegativeIntegers, bounds=(0,1))

def obj_expression(model):
    #return pyo.summation(m.c, m.x)
    return Upsetspyo(matrix, model.r, unique, N, d)

model.OBJ = pyo.Objective(rule=obj_expression)

# the next line creates one constraint for each member of the set model.I
model.constraints = pyo.ConstraintList()
for i in range(N):
    print("constraint: " + str(i))
    model.constraints.add(expr = sum(model.r[i,j] for j in range(N)) == 1)
    model.constraints.add(expr = sum(model.r[j,i] for j in range(N)) == 1)
print(model.constraints)

solver = pyo.SolverFactory('cplex')
solver.options['timelimit'] = 6000
results = solver.solve(model, tee=True)
model.solutions.load_from(results)
original_stdout = sys.stdout
f = open('Testing.txt', 'w+')
sys.stdout = f
#model.r.display()
model.display()
sys.stdout = original_stdout
f.close()
#model.r.display()
model.display()

