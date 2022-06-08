import time
import pprint
import math
import random
import numpy as np
import datetime as dt
from scipy import linalg
from scipy.linalg import solve
from scipy.sparse.linalg import eigs
from scipy.sparse import dok_matrix
from dateutil.relativedelta import relativedelta
import pyomo.environ as pyo
from pyomo.common.tempfiles import TempfileManager
TempfileManager.tempdir = "D:/Memoire"
import matplotlib.pyplot as plt
import sys
import seaborn as sns

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def Gauss(x):
    return np.exp(-x*x/5)

def dGauss(x):
    return -2 * x * Gauss(x)/5

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

def ReadYear(file, year):
    
    f = open(file + str(year) + ".csv", "r", encoding="utf-8")
    line = f.readline()
    line = f.readline()
    line = line.split(",")
    # date, winner name, loser name
    line = [line[10], line[18], "1", "0", line[4], line[25]]
    read = [line]
    
    while f:
        line = f.readline()
        line = line.split(",")
        if line == [""]:
            break
        line = [line[10], line[18], "1", "0", line[4], line[25]]
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

def ReadResults(file, unique, N):
    rank = [None] * N
    f = open(file, "r")
    while f:
        line = f.readline()
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        print("line1:")
        print(line)

        if line != "":
            if line[0] == 'K':
                break
    
    while f:
        line = f.readline()
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        if line == "":
            break
        print("line2:")
        print(line)
        line = line.split(":")
        line0 = line[0].split(",")[0]
        line0 = line0.split("(")[1]
        line1 = line[0].split(",")[1]
        line1 = line1.split(")")[0]
        line2 = line[2].split(".")[0]
        if int(line2) == 1:
            rank[int(line1)] = unique[int(line0)]
    f.close()
    return rank

def ReadResults2(file, unique, N):
    rank = []
    r = np.zeros([N,N])
    f = open(file, "r")
    while f:
        line = f.readline()
        
        line = line.replace(" ", "")
        line = line.replace("\n", "")

        if line == "Variable:" or line == "":
            break
    
    while f:
        line = f.readline()
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        if line == "Constraint:Novalues" or line == "":
            break
        line = line.split("[")
        line = line[1].split("]")[0]
        line2 = f.readline()
        line2 = line2.replace(" ", "")
        line2 = line2.replace("\n", "")
        line2 = line2.split(":")[1]
        rank.append([unique[int(line)], line2])
    f.close()
    return rank, r

def ReadResults3(file, unique, N):
    rank = []
    r = np.zeros([N,N])
    f = open(file, "r")
    while f:
        line = f.readline()
        
        line = line.replace(" ", "")
        line = line.replace("\n", "")

        if line[0] == "(" or line == "":
            break

    while f:
        line = f.readline()
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        if line == "":
            break
        line = line.split(":")
        line2 = line[2]
        line2 = line2.split(".")[0]
        line = line[0]
        line = line.split("(")[1]
        line = line.split(")")[0]
        line = line.split(",")
        if int(line2) == 1:
            r[int(line[0]),int(line[1])] = 1
            rank.append([unique[int(line[0])], line[1]])
    f.close()
    return rank, r

def Countries(data):
    read = np.array(data)
    countries = read[:, [0,1]]
    return np.unique(countries)
"""
def Clubs(data):
    read = np.array(data)
    countries = read[:, [2,3]]
    return np.unique(countries)
"""
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

def MatrixRank(data, rank, N):
    matrix = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(d):
        if data[i][2] > data[i][3]:
            matrix[np.where(rank == data[i][0])[0][0]][np.where(rank == data[i][1])[0][0]] += 1
        if data[i][2] < data[i][3]:
            matrix[np.where(rank == data[i][1])[0][0]][np.where(rank == data[i][0])[0][0]] += 1
    return matrix

def MatchTableImportance(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(d):
        if data[i][2] > data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 5
            elif (data[i][4] == "FIFA World Cup qualification"): # or EURO or concaf, ...
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 3
            elif data[i][4] == "Friendly":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
            else:
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 3
        if data[i][2] < data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 5
            elif data[i][4] == "FIFA World Cup qualification":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 3
            elif data[i][4] == "Friendly":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 1
            else:
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 3
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
    matrix = matrix.astype(np.float32)
    for i in range(N):
        s = np.sum(matrix[i])
        if s != 0:
            matrix[i] = matrix[i]/float(s)
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

def NoWin(matrix, unique, N):
    i = 0
    countries = []
    for i in range(len(unique)):
        if np.sum(matrix[i]) == 0:
            countries.append(unique[i])
    
    if len(countries) == 0:
        return None
    return countries

def OnlyWin(matrix, unique, N):
    i = 0
    countries = []
    for i in range(len(unique)):
        if np.sum(matrix[:,i]) == 0:
            countries.append(unique[i])
    
    if len(countries) == 0:
        return None
    return countries

def RemoveNoWin(matrix, m, unique, N):
    countries = []
    last = []
    unique = np.array(unique)
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

def RemoveNoMatch(matrix, m, unique, N):
    countries = []
    unique = np.array(unique)
    i = 0
    while i < N:
        if sumlc(matrix, i, i) == 0:
            matrix = np.delete(matrix, i, axis = 0)
            matrix = np.delete(matrix, i, axis = 1)
            m = np.delete(m, i, axis = 0)
            m = np.delete(m, i, axis = 1)
            unique = np.delete(unique, i)
            N -= 1
        i += 1
    return matrix, m, unique, N

def RemoveOnlyWin(matrix, m, unique, N):
    countries = []
    first = []
    unique = np.array(unique)
    while True:
        countries = OnlyWin(matrix, unique, N)
        if countries is None:
            break
        else:
            first.append(countries)
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
    return matrix, m, unique, N, first

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

def Upsets(matrix, rank, unique, N, d):
    upsets = 0
    for i in range(N):
        for j in range(N):
            if i > j:
                #print(unique)
                #print(rank[i])
                upsets += matrix[np.where(unique == rank[i])[0][0]][np.where(unique == rank[j])[0][0]]
    return upsets

def UpsetsByCountries(matrix, rank, unique, N, d):
    upsets = []
    upset = 0
    old = 0
    for i in range(N):
        upsets.append([])
        old = upset
        for j in range(N):
            if i > j:
                upornot = matrix[np.where(unique == rank[i])[0][0]][np.where(unique == rank[j])[0][0]]
                if upornot > 0:
                    upsets[i].append(j)
                    upset += upornot

        #upsets[i] = upset - old
    return upsets, upset

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

def ImproveRanking(m, rank, unique, N, d):
    UpsetsByC, upsets = UpsetsByCountries(m, rank, unique, N, d)
    new_rank = rank.copy()
    i = 0
    progress = 0
    while i < N:
        if i > progress:
            progress = i
            print(str(i) +"/"+ str(N))
        
        if len(UpsetsByC[i]) > 0:
            new_rank.insert(UpsetsByC[i][-1], new_rank.pop(i))
            new_upsets = Upsets(m, new_rank, unique, N, d)
            if new_upsets < upsets:
                rank = new_rank.copy()
                upsets = new_upsets
                i = 0
            else:
                new_rank = rank.copy()
        i += 1

    return rank, upsets

def ImproveRanking2(m, rank, unique, N, d):
    UpsetsByC, upsets = UpsetsByCountries(m, rank, unique, N, d)
    new_rank = rank.copy()
    change = 1
    while change != 0:
        old = upsets
        i = 0
        progress = 0
        while i < N:
            if i > progress:
                progress = i
        
            if len(UpsetsByC[i]) > 0:
                new_rank.insert(UpsetsByC[i][-1], new_rank.pop(i))
                new_upsets = Upsets(m, new_rank, unique, N, d)
                if new_upsets < upsets:
                    rank = new_rank.copy()
                    upsets = new_upsets
                else:
                    new_rank = rank.copy()
            i += 1
        change = old - upsets


    return rank, upsets

def Score(matrix, score_rank, unique, N):
    score = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if matrix[i][j] > 0:
                score[i] += sigmoid(score_rank[i] - score_rank[j]) * matrix[i][j]
            if matrix[j][i] > 0:
                score[i] -= sigmoid(score_rank[i] - score_rank[j]) * matrix[j][i]
            if draws[i][j] + draws[j][i] > 0:
                score[i] += dGauss(score_rank[j] - score_rank[i]) * (draws[i][j] + draws[j][i])
            
    #print(score)
    return score

def MatchTableDraws(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    draws = [[0]*N for i in range(N)]
    d = len(data)
    unique = np.array(unique)
    for i in range(d):
        """
        print(unique)
        print(data[i][0])
        print(data[i][1])
        print(np.where(unique == data[i][0])[0][0])
        print(np.where(unique == data[i][1])[0][0])
        """

        if data[i][2] > data[i][3]:
            matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
        if data[i][2] == data[i][3]:
            draws[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
        if data[i][2] < data[i][3]:
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 1
    return matrix, draws

def MatchTableDrawsSmall(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    draws = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(d):
        if data[i][2] > data[i][3]:
            matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1/d
        if data[i][2] == data[i][3]:
            draws[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1/d
        if data[i][2] < data[i][3]:
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 1/d
    return matrix, draws

def MatchTableDrawsWR(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    draws = [[0]*N for i in range(N)]
    matches = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(len(data)):
        if data[i][2] > data[i][3]:
            matrix, matches = WR(np.where(unique == data[i][0])[0][0], np.where(unique == data[i][1])[0][0], matrix, matches, 1)
        if data[i][2] == data[i][3]:
            draws, matches = WR(np.where(unique == data[i][0])[0][0], np.where(unique == data[i][1])[0][0], matrix, matches, 1)
        if data[i][2] < data[i][3]:
            matrix, matches = WR(np.where(unique == data[i][1])[0][0], np.where(unique == data[i][0])[0][0], matrix, matches, 1)
    return matrix, draws

def MatchTableDrawsImportance(data, unique, N, imp):
    matrix = [[0]*N for i in range(N)]
    draws = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(d):
        if data[i][2] > data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += imp[0]*imp[1]
            elif data[i][4] == "Friendly":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
            else:
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += imp[1]
        if data[i][2] == data[i][3]:
            if data[i][4] == "FIFA World Cup":
                draws[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += imp[0]*imp[1]
            elif data[i][4] == "Friendly":
                draws[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
            else:
                draws[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += imp[1]
        if data[i][2] < data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += imp[0]*imp[1]
            elif data[i][4] == "Friendly":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 1
            else:
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += imp[1]
    
    return matrix, draws

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

def HasANullTerm(matrix, N):
    for i in range(N):
        for j in range(N):
            if matrix[i][j] == 0:
                return True
            
    return False

def IndirectMatrix(matrix, unique, N):
    new_matrix = matrix.copy()
    n_matrix = matrix.copy()
    for t in range(N):
        print(t)
        for i in range(N):
            for j in range(N):
                new_matrix[i] += n_matrix[i][j] * n_matrix[j]
        if HasANullTerm(new_matrix, N):
            n_matrix = new_matrix.copy()
        else:
            break

    return n_matrix

def ELO(matrix, rank, unique, N, C):
    new_rank = rank.copy()
    for i in range(N):
        j = 0
        while j < N:
            k = 0
            R1 = math.pow(10, (rank[i]/400))
            R2 = math.pow(10, (rank[j]/400))
            while k < matrix[i][j]:
                new_rank[i] += C * (1-(R1/(R1+R2)))
                new_rank[j] += C * (0-(R2/(R1+R2)))
                k += 1
            j += 1
    
    return new_rank

def extractTesting(file, unique):
    unique1 = []
    unique2 = []
    f = open(file, "r")
    while f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()

        break

    while f:
        line = f.readline()
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        if line == "":
            break
        line = line.split(":")
        line2 = line[2]
        line2 = line2.split(".")[0]
        line = line[0]
        if int(line2) > 0:
            unique2.append(unique[int(line)])
        else:
            unique1.append(unique[int(line)])
    f.close()
    return unique1, unique2

def Clusters(matrix, r, unique, N, d):
    upsets = 0
    j = 0
    for i in range(N):
        j = 0
        while j < N:
            upsets += matrix[i][j] * (r[j]-r[i]) * r[j]
            j += 1
    #print(upsets)
    return upsets

def Minimum(matrix, unique):
    minim = 0
    for i in range(len(unique)):
        j = i + 1
        while j < len(unique):
            if matrix[i][j] > 0 and matrix[j][i] > 0:
                minim += min(matrix[j][i], matrix[i][j])
            j += 1
    return minim

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

def KeepOnlyGames(data, countries, d):
    i = 0
    new_d = 0
    new_data = []
    while i < d:
        if data[i][0] in countries and data[i][1] in countries:
            new_data.append(data[i])
            new_d += 1
        i += 1

    return new_data, new_d

def UpsetsPyo(matrix, r, unique, N, d):
    upsets = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    if k >= l:
                        if matrix[i][j] > 0:
                            upsets += (r[i,k] * r[j,l]) * matrix[i][j]                     
                        
    return upsets

def Score(matrix, draws, score_rank, unique, N):
    score = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if matrix[i][j] > 0:
                score[i] += sigmoid(score_rank[i] - score_rank[j]) * matrix[i][j]
            if matrix[j][i] > 0:
                score[i] -= sigmoid(score_rank[i] - score_rank[j]) * matrix[j][i]
            #if draws[i][j] + draws[j][i] > 0:
            #    score[i] += dGauss(score_rank[j] - score_rank[i]) * (draws[i][j] + draws[j][i])
            
    #print(score)
    return score

def Score_deriv(matrix, draws, score_rank, unique, N):
    score = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if matrix[i][j] > 0:
                score[i] += derivative(score_rank[i] - score_rank[j]) * matrix[i][j]
            if matrix[j][i] > 0:
                score[i] -= derivative(score_rank[i] - score_rank[j]) * matrix[j][i]
            #if draws[i][j] + draws[j][i] > 0:
            #    score[i] += dGauss(score_rank[j] - score_rank[i]) * (draws[i][j] + draws[j][i])/2
            
    #print(score)
    return score

def new_Score(matrix, draws, score_rank, unique, new_unique):
    N = len(unique)
    new_N = len(new_unique)

    score = np.zeros(N)
    for i in range(N):
        if unique[i] in new_unique:
            for j in range(N):
                a = np.where(new_unique == unique[i])[0][0]
                if unique[j] in new_unique:
                    b = np.where(new_unique == unique[j])[0][0]
                    #print(a)
                    #print(b)
                    if matrix[a][b] > 0:
                        score[i] += sigmoid(score_rank[a] - score_rank[b]) * matrix[a][b]
                    if matrix[b][a] > 0:
                        score[i] -= sigmoid(score_rank[a] - score_rank[b]) * matrix[b][a]
                    #if draws[a][b] + draws[b][a] > 0:
                    #    score[i] += dGauss(score_rank[b] - score_rank[a]) * (draws[a][b] + draws[b][a])/2

    #print(score)
    return score

def new_Score_deriv(matrix, draws, score_rank, unique, new_unique):
    N = len(unique)
    new_N = len(new_unique)

    score = np.zeros(N)
    for i in range(N):
        if unique[i] in new_unique:
            for j in range(N):
                a = np.where(new_unique == unique[i])[0][0]
                if unique[j] in new_unique:
                    b = np.where(new_unique == unique[j])[0][0]
                    #print(a)
                    #print(b)
                    
                    if matrix[a][b] > 0:
                        score[i] += derivative(score_rank[a] - score_rank[b]) * matrix[a][b]
                    if matrix[b][a] > 0:
                        score[i] -= derivative(score_rank[a] - score_rank[b]) * matrix[b][a]
                    #if draws[a][b] + draws[b][a] > 0:
                    #    score[i] += dGauss(score_rank[b] - score_rank[a]) * (draws[a][b] + draws[b][a])/2

    #print(score)
    return score

def SetScore(last, matrix, unique, score, N, d):
    """
    for i in range(len(last)):
        for j in range(len(last[i])):
            #score[np.where(unique == last[i][j])] = i*2.5+2.5
            score[np.where(unique == last[i][j])] = i+1
            """

    for i in range(len(unique)):
        if score[i] == 0:
            score[i] = sum(matrix[i])*50*N/d
            #score[i] = i
            #print(score[i])
    return score

def UpdateScore(last, unique, score):
    """
    for i in range(len(last)):
        for j in range(len(last[i])):
            #score[np.where(unique == last[i][j])] = i*2.5+2.5
            score[np.where(unique == last[i][j])] = i+1
    """
    minimum = min(score)
    score -= minimum
    maximum = max(score)
    score = score*100/maximum
    
    return score

def PickData(read, unique):
    new_read = []
    for i in range(len(read)):
        rand = random.randint(0,1)
        if rand == 0 and (read[i][1] in unique or read[i][2] in unique):
            new_read.append(read[i])

    return np.array(new_read)

def NoWinDraws(matrix, draws, unique, N):
    i = 0
    countries = []
    for i in range(len(unique)):
        if np.sum(matrix[i]) == 0 and np.sum(draws[i]) == 0 and np.sum(draws[:,i]) == 0:
            countries.append(unique[i])
    
    if len(countries) == 0:
        return None
    return countries

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

def Armijo(matrix, draws, unique, new_unique, last, score_rank, f, grad, new_f, d, step, rho, b1):
    N = len(unique)

    while condition1(f, new_f, grad, d, step, b1):
        #print("a: " + str(step))
        step = step * rho
        new_score_rank = UpdateScore(last, unique, score_rank + step * grad)
        new_f = sum(new_Score(matrix, draws, new_score_rank, unique, new_unique))
        
    return step, new_f

def  Wolfe(matrix, draws, unique, new_unique, last, score_rank, grad, new_grad, d, step, rho, b2):
    N = len(unique)
    while condition2(grad, new_grad, d, step, b2):
        #print("w: " + str(step))
        step = step/rho
        new_score_rank = UpdateScore(last, unique, score_rank + step * grad)
        new_grad = new_Score_deriv(matrix, draws, new_score_rank, unique, new_unique)

    return step, new_grad

def condition1(f, new_f, grad, d, step, b1):
    #print("One: " + str(new_phi < phi + b1 * step))
    return new_f <= f + b1 * np.dot(grad, d) and step > 0.001/np.linalg.norm(grad) #np.linalg.norm(grad) # np.dot(grad, d) # sum(np.absolute(grad))

def condition2(grad, new_grad, d, step, b2):
    #print("Two: " + str(new_grad/grad < b2))
    #print(np.dot(new_grad, d))
    #print(np.dot(grad, d))

    return np.dot(new_grad, d)/np.dot(grad, d) <= b2 and step < 10/np.linalg.norm(grad)

def conditions(f, grad, new_f, new_grad, d, step, b1, b2):
    return condition1(f, new_f, grad, d, step, b1) or condition2(grad, new_grad, d, step, b2)

def get_rank(vec, unique):
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    rank = [tple[0] for tple in ranked]
    return rank


def RandomRanking(N):
    rank = np.zeros(N)
    for i in range(N):
        rank[i] = random.randint(0,100)
        
    return rank
