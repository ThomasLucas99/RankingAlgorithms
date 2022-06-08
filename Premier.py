# Parser for extracting the correct information from the database
from __future__ import division

import time
import pprint
import math
import numpy as np
import datetime as dt
from scipy import linalg
from scipy.linalg import solve
from scipy.sparse.linalg import eigs
from scipy.sparse import dok_matrix
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def integral(x):
    return np.log(np.absolute(np.cosh(x)))

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

def ReadResults(file, unique, N):
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
        line = line[1].split(",")
        line[1] = line[1].split("]")[0]
        line2 = f.readline()
        r[int(line[0]),int(line[1])] = 1
        rank.append([unique[int(line[0])], line[1]])
    f.close()
    return rank, r

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
    countries = read[:, [2,3]]
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
            elif data[i][4] == "FIFA World Cup qualification":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 3
            elif data[i][4] == "Friendly":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
            else:
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 3
        if data[i][2] < data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 4
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
    old = 0
    for i in range(N):
        old = upsets
        for j in range(N):
            if i >= j:
                upsets += matrix[np.where(unique == rank[i])[0][0]][np.where(unique == rank[j])[0][0]]
    return upsets

def UpsetsByCountries(matrix, r, unique, N, d):
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

def ImproveRanking(matrix, rank, unique, N, d):
    UpsetsByC, upsets = UpsetsByCountries(m, rank, unique, N, d)
    new_rank = rank.copy()
    i = 0
    progress = 0
    while i < N:
        if i > progress:
            progress = i
            print(str(i) +"/"+ str(N))
        
        if len(UpsetsByC[i]) > 0:
            #print(rank)
            new_rank.insert(UpsetsByC[i][-1], new_rank.pop(i))
            new_upsets = Upsets(m, new_rank, unique, N, d)
            #print(new_upsets)
            #print(upsets)
            #print(rank)
            if new_upsets < upsets:
                rank = new_rank.copy()
                upsets = new_upsets
                i = 0
            else:
                new_rank = rank.copy()
        i += 1

    return rank, upsets

def ImproveRanking2(matrix, rank, unique, N, d):
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
                #print(str(i) +"/"+ str(N) + " " + str(change))
        
            if len(UpsetsByC[i]) > 0:
                #print(rank)
                new_rank.insert(UpsetsByC[i][-1], new_rank.pop(i))
                new_upsets = Upsets(m, new_rank, unique, N, d)
                #print(new_upsets)
                #print(upsets)
                #print(rank)
                if new_upsets < upsets:
                    rank = new_rank.copy()
                    upsets = new_upsets
                else:
                    new_rank = rank.copy()
            i += 1
        change = old - upsets


    return rank, upsets

def MatchTableDraws(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    draws = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(d):
        if data[i][2] > data[i][3]:
            matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
        if data[i][2] == data[i][3]:
            draws[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
        if data[i][2] < data[i][3]:
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 1
    return matrix, draws

def MatchTableDrawsImportance(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    draws = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(d):
        if data[i][2] > data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 5
            elif data[i][4] == "FIFA World Cup qualification":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 3
            elif data[i][4] == "Friendly":
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
            else:
                matrix[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 3
        if data[i][2] == data[i][3]:
            if data[i][4] == "FIFA World Cup":
                draws[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 5
            elif data[i][4] == "FIFA World Cup qualification":
                draws[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 3
            elif data[i][4] == "Friendly":
                draws[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 1
            else:
                draws[np.where(unique == data[i][0])[0][0]][np.where(unique == data[i][1])[0][0]] += 3
        if data[i][2] < data[i][3]:
            if data[i][4] == "FIFA World Cup":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 5
            elif data[i][4] == "FIFA World Cup qualification":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 3
            elif data[i][4] == "Friendly":
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 1
            else:
                matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][0])[0][0]] += 3
    
    return matrix, draws

def Minimum(matrix, unique):
    minim = 0
    for i in range(len(unique)):
        j = i + 1
        while j < len(unique):
            if matrix[i][j] > 0 and matrix[j][i] > 0:
                minim += min(matrix[j][i], matrix[i][j])
            j += 1
    
    for i in range(len(unique)):
        j = i + 1
        m = 1000
        while j < len(unique):
            k = j + 1
            while k < len(unique):
                if matrix[i][j] > 0 and matrix[j][k] > 0 and matrix[k][i] > 0:
                    new_m = min([matrix[i][j], matrix[j][k], matrix[k][i]])
                    if new_m < m:
                        m = new_m
                k += 1
            j += 1
        if m != 1000:
            minim += m
    
    return minim

def Score(matrix, draws, score_rank, unique, N):
    score = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if matrix[i][j] > 0:
                score[i] += sigmoid(score_rank[i] - score_rank[j]) * matrix[i][j]
            if matrix[j][i] > 0:
                score[i] += sigmoid(score_rank[j] - score_rank[i]) * matrix[j][i]
            if draws[i][j] + draws[j][i] > 0:
                score[i] += Gauss(score_rank[j] - score_rank[i]) * (draws[i][j] + draws[j][i])
            
    #print(score)
    return score

def Score_deriv(matrix, draws, score_rank, unique, N):
    score = np.zeros(N)
    for i in range(N):
        for j in range(N):
            #print(str(i) + " " + str(j))
            #print(score)
            #print(score_rank)
            if matrix[i][j] > 0:
                score[i] += derivative(score_rank[i] - score_rank[j]) * matrix[i][j]
                #print(score_rank[i] - score_rank[j])
                #print("1 :" + str(score))
            if matrix[j][i] > 0:
                score[i] -= derivative(score_rank[i] - score_rank[j]) * matrix[j][i]
                #print("2 :" + str(score))
            if draws[i][j] + draws[j][i] > 0:
                score[i] += dGauss(score_rank[i] - score_rank[j]) * (draws[i][j] + draws[j][i])
                #print("3 :" + str(score))
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
                    if draws[a][b] + draws[b][a] > 0:
                        score[i] += dGauss(score_rank[b] - score_rank[a]) * (draws[a][b] + draws[b][a])/2

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
                    if draws[a][b] + draws[b][a] > 0:
                        score[i] += dGauss(score_rank[b] - score_rank[a]) * (draws[a][b] + draws[b][a])/2

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
            score[i] = sum(matrix[i])*5*N/d
            #score[i] = random.randint(0, 100)
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
    score = score*10/maximum
    
    return score

def Armijo(matrix, draws, unique, score_rank, f, grad, new_f, d, step, rho, b1):
    N = len(unique)

    while condition1(f, new_f, grad, d, step, b1):
        print("a: " + str(step))
        step = step * rho
        new_score_rank = UpdateScore(last, unique, score_rank + step * grad)
        new_f = sum(Score(matrix, draws, new_score_rank, unique, N))
        
    return step, new_f

def  Wolfe(matrix, draws, unique, score_rank, grad, new_grad, step, rho, b2):
    N = len(unique)
    while condition2(grad, new_grad, step, b2):
        print("w: " + str(step))
        step = step/rho
        new_score_rank = UpdateScore(last, unique, score_rank + step * grad)
        new_grad = Score_deriv(matrix, draws, new_score_rank, unique, N)

    return step, new_grad

def condition1(f, new_f, grad, d, step, b1):
    #print("One: " + str(new_phi < phi + b1 * step))
    return new_f <= f + b1 * np.dot(grad, d) and step > 0.01/np.dot(grad, d) #np.linalg.norm(grad) # np.dot(grad, d) # sum(np.absolute(grad))

def condition2(grad, new_grad, step, b2):
    #print("Two: " + str(new_grad/grad < b2))
    #print(np.dot(new_grad, d))
    #print(np.dot(grad, d))

    return np.dot(new_grad, d)/np.dot(grad, d) <= b2 and step < 1/np.dot(grad, d)

def conditions(f, grad, new_f, new_grad, d, step, b1, b2):
    return condition1(f, new_f, grad, d, step, b1) or condition2(grad, new_grad, step, b2)

def get_rank(vec, unique):
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    rank = [tple[0] for tple in ranked]
    return rank

def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))
     
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


file = "Data/Premier/2018_2019.csv"
start = dt.datetime(1900, 1, 1)
date = dt.datetime(2014, 1, 1)
#date = dt.datetime(2020, 5, 27)
end = dt.datetime(2021, 1, 1)
#end = dt.datetime(2021, 5, 27)

#read = np.array(ReadDataFromDate(file, date))
#read = np.array(ReadDataFromWC(file, date))
#read = np.array(ReadDataFromWCQ(file, date))
#read = np.array(ReadDataDates(file, date, end))
#read = np.array(ReadDataDatesWC(file, date, end))
#read = read[:22]
#read = read[:44]
read = np.array(ReadPremier(file))
print(read)

unique = Countries(read)
N = len(unique)
data = read[:, 2:6]
print(data)
ldata = len(data)
print(ldata)
m, draws = np.array(MatchTableDraws(data, unique, N))
print(m)
msum = np.sum(m)
minim = Minimum(m, unique)
print("m: " + str(minim))

#matrix = np.array(MatchTable(data, unique, N)).astype(np.float64)
matrix, draws = np.array(MatchTableDraws(data, unique, N))
"""
matrix, m, unique, N, last = RemoveNoWin(matrix, m, unique, N)
print(m)
flat_last = Flat(last)
data, ldata, removed_games = RemoveGames(data, flat_last, ldata)
"""
#matrix = matrix + np.diag([1 for i in range(N)])
#matrix = MatrixNormalizationLine(matrix, N)
#matrix = MatrixNormalizationCol(matrix, N)
#matrix = MatrixNormalization(matrix, N)
print(msum)

#new_data = MakeDataGreat(data, unique)

csv = "Matrix.csv"

WriteCSV(csv, matrix, unique, N)
"""
#Trying to compare real ranking to other ones

# Real ranking
r, unique, m, matrix, N, data, ldata = ExtractRankingFile("fifa_ranking.csv", unique, m, matrix, date, data, ldata)

print(r[0:10])
print("Upsets")
print(Upsets(m, r, unique, N, ldata))

"""
last = []

# 18-19
real_rank = ['Man City', 'Liverpool', 'Chelsea', 'Tottenham', 'Arsenal', 'Man United', 'Wolves', 'Everton', 'Leicester', 'West Ham', 'Watford', 'Crystal Palace', 'Newcastle', 'Bournemouth', 'Burnley', 'Southampton', 'Brighton', 'Cardiff', 'Fulham', 'Huddersfield']
real_score = [98, 97, 72, 71, 70, 66, 57, 54, 52, 52, 50, 49, 45, 45, 40, 39, 36, 34, 26, 16]
real_unique = ['Arsenal', 'Bournemouth', 'Brighton', 'Burnley', 'Cardiff', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Huddersfield', 'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Southampton', 'Tottenham', 'Watford', 'West Ham', 'Wolves']
real_score = [70, 45, 36, 40, 34, 72, 49, 54, 26, 16, 52, 97, 98, 66, 45, 39, 71, 50, 52, 57]

"""
# 17-18
real_rank = ['Man City', 'Man United', 'Tottenham', 'Liverpool', 'Chelsea', 'Arsenal', 'Burnley', 'Everton', 'Leicester', 'Newcastle', 'Crystal Palace', 'Bournemouth', 'West Ham', 'Watford', 'Brighton', 'Huddersfield', 'Southampton', 'Swansea', 'Stoke', 'West Brom']
real_score = [100, 81, 77, 75, 70, 63, 54, 49, 47, 44, 44, 44, 42, 41, 40, 37, 36, 33, 33, 31]
real_unique = ['Arsenal', 'Bournemouth', 'Brighton', 'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Huddersfield', 'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Southampton', 'Stoke', 'Swansea', 'Tottenham', 'Watford', 'West Brom', 'West Ham']
real_score = [63, 44, 40, 54, 70, 44, 49, 37, 47, 75, 100, 81, 44, 36, 33, 33, 77, 41, 31, 42]
"""
real_score = np.array(real_score)

print(unique)

print("Real")
print(Upsets(m, real_rank, unique, N, ldata))
#real_score = real_score/5
real_score = UpdateScore(last, unique, real_score)
print(real_rank)
print(real_score)
print(sum(Score(matrix, draws, real_score, unique, N)))

print("ELO")
rank = np.ones(N)*1000
rank_team = np.zeros(N)
for t in range(1000):
    rank = ELO(matrix, rank, unique, N, 1)
    vec = rank
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    rank_team = [tple[0] for tple in ranked]
    rank_score = [tple[1] for tple in ranked]
    print(rank_team[0:10])
    print(rank_score[0:10])
    #print(ranked)
    upsets = Upsets(m, rank_team, unique, N, ldata)
    print("Upsets: " + str(upsets))

print("Kendall")
print(normalised_kendall_tau_distance(rank_team, real_rank)*20*19)

gradient = True
best_upsets = 100000
step = 1
rho = 0.9 
b1 = 0.1
b2 = 0.2
phi = 1
init_value = np.zeros(N)
T = 1000
rating = 0
best_rating = 0

scores = np.zeros(T)
upsets_plt = np.zeros(T)
g_plt = np.zeros(T)
"""
for i in range(N):
    init_value[i] = random.randint(0,100)
"""
ranking = []
new_gradient = []
if gradient == True:
    j = 0
    score_rank = np.zeros(N)
    #for i in range(N):
    #    score_rank[i] = random.randint(0,100)
    # get the teams that do not have wins or draws
    new_matrix, new_m, new_unique, new_N, last = RemoveNoWinDraws(matrix, draws, m, unique, N)
    #print(last)
    flat_last = Flat(last)
    # set the last's score
    score_rank = SetScore(last, matrix, unique, score_rank, N, ldata)
    score_rank = UpdateScore(last, unique, score_rank)
    vec = score_rank
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    rank_score = [tple[1] for tple in ranked]
    print(rank_score)
    print(Score_deriv(matrix, draws, score_rank, unique, N))
    step = 100/np.linalg.norm(Score_deriv(matrix, draws, score_rank, unique, N))
    print("step: " + str(step))
    #print(score_rank)
    # set the score of other teams
    #rank_score = np.ones(N) * init_value
    rank_score = init_value
    rank_score = rank_score.tolist()

    vec = score_rank
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    rank = [tple[0] for tple in ranked]
    print(Upsets(m, rank, unique, N, ldata))
    print(sum(Score(matrix, draws, score_rank, unique, N)))

    #while rank_score[0] < 10 and rank_score[-len(flat_last)-1] > len(last)+1:
    for t in range(T):
    #take random sample of matches
        
        #new_read = PickData(read, unique)
        new_read = read
        #print(new_read)
        new_unique = Countries(new_read)
        new_data = new_read[:, 2:6]
        new_N = len(new_unique)
        new_d = len(new_data)
        new_matrix, new_draws = np.array(MatchTableDraws(new_data, new_unique, new_N))
        
        # compute gradient
        #new_gradient = new_Score(new_matrix, new_draws, score_rank, new_unique, unique)
        print("first")
        #print(new_matrix)
        #print(score_rank)
        new_gradient = Score_deriv(matrix, new_draws, score_rank, unique, N)
        #print(new_gradient)
        print("gradient: " + str(np.linalg.norm(new_gradient)))
        #gradient = Score(matrix, draws, score_rank, unique, N)
        #print(score_rank)
        # apply the gradient
        #print("vector")
        #print(np.absolute(new_gradient))
        #score_rank += (new_gradient/max(np.absolute(new_gradient)))/1000
        
        # update the step
        
        f = sum(Score(matrix, new_draws, score_rank, unique, N))
        print("score: " + str(f))
        grad = new_gradient
        d = np.sign(grad)
        new_score_rank = UpdateScore(last, unique, score_rank + step * grad)
        new_f = sum(Score(matrix, new_draws, new_score_rank, unique, N))
        new_grad = Score_deriv(matrix, new_draws, new_score_rank, unique, N)
        
        while conditions(f, grad, new_f, new_grad, d, step, b1, b2):
            step, new_f = Armijo(matrix, draws, unique, score_rank, f, grad, new_f, d, step, rho, b1)
            step, new_grad = Wolfe(matrix, draws, unique, score_rank, grad, new_grad, step, rho, b2)
        
        score_rank += new_gradient*step
        #print("gradient*step: " + str(np.linalg.norm(new_gradient)*step))
        # set back the last's scores
        score_rank = UpdateScore(last, unique, score_rank)
        #print(score_rank)
        #print(score_rank)
        vec = score_rank
        sorted_indices = vec.argsort()
        ranked = [(unique[i], vec[i]) for i in sorted_indices]
        ranked.reverse()
        rank = [tple[0] for tple in ranked]
        rank_score = [tple[1] for tple in ranked]
        #print(rank)
        #print(rank_score)
        upsets = Upsets(m, rank, unique, N, ldata)

        upsets_plt[t] = upsets
        s1 = UpdateScore(last, unique, vec)
        scores[t] = sum(Score(matrix, draws, s1, unique, N))
        g_plt[t] = np.linalg.norm(new_grad)
        """
        if upsets < best_upsets:
            best_upsets = upsets
            ranking = rank
            print("j: " + str(j))
            print("upsets: " + str(upsets))
            print(ranking[:10])
            r_score = rank_score[:10]
            #for i in range(10):
            #    r_score[i] = int(r_score[i]*20)
            print(r_score)
            """
        if scores[t] > rating:
            best_score = scores[t]
            ranking = rank
            print("j: " + str(j))
            print("upsets: " + str(upsets))
            print(ranking[:10])
            r_score = rank_score[:10]
            #for i in range(10):
            #    r_score[i] = int(r_score[i]*20)
            print(r_score)

        #print(rank_score[0])
        #print(rank_score[-len(flat_last)-1])
        j += 1
    
    print(rank[:N])
    print(new_gradient[:N])
    r_score = rank_score[:N]
    print(r_score)
    upsets = Upsets(m, rank, unique, N, ldata)
    print(upsets)
    """
    startTime = time.time()
    print("Improving the ranking")
    new_rank, upsets = ImproveRanking(matrix, rank, unique, N, ldata)
    print(time.time() - startTime)
    print(new_rank)
    print(upsets)
    """
    print("Kendall")
    print(normalised_kendall_tau_distance(rank, real_rank)*20*19)
    
    #s1 = UpdateScore(last, unique, vec)
    #print(sum(Score(matrix, s1, unique, N)))

    plt.plot(upsets_plt)
    plt.show()
    plt.plot(scores)
    plt.show()
    plt.plot(g_plt)
    plt.show()
    
else:
    matrix, m, unique, N, last = RemoveNoWin(matrix, m, unique, N)
    flat_last = Flat(last)
    data, ldata, removed_games = RemoveGames(data, flat_last, ldata)

# Colley
print(N)
print("Colley")
b = np.zeros([N])
C = np.zeros([N, N])
for i in range(N):
    for j in range(N):
        if i == j:
            C[i][j] = 2 + sumlc(matrix, i, j)
        else:
            C[i][j] = -matrix[i][j] - matrix[j][i]
for i in range(N):
    b[i]= 1 + (np.sum(matrix[i]) - np.sum(matrix[:, i]))/2
s = solve(C,b)
s1 = s
z = [x for _,x in sorted(zip(s,unique))]
rank = np.flip(z)
print(rank[0:10])
print("Upsets")
upsets = Upsets(m, rank, unique, N, ldata)
print(upsets)
print(upsets/msum)

startTime = time.time()
#rank, upsets = ImproveRanking2(matrix, rank.tolist(), unique, N, ldata)
print(time.time() - startTime)
print(upsets)
print(rank)
s1 = UpdateScore(last, unique, s1)
print(sum(Score(matrix, draws, s1, unique, N)))

print("Kendall")
print(normalised_kendall_tau_distance(rank, real_rank)*20*19)

rank = np.array(rank)
mr = MatrixRank(data, rank, N)
csv = "MatrixRank.csv"

WriteCSV(csv, mr, rank, N)


# Eigen values
print("Eigen")
w, v = np.linalg.eig(matrix)
vec = np.absolute(v[:, np.argmax(w)])
sorted_indices = vec.argsort()
ranked = [(unique[i], vec[i]) for i in sorted_indices]
ranked.reverse()
rank = [tple[0] for tple in ranked]
print(rank[0:10])
print("Upsets")
print(Upsets(m, rank, unique, N, ldata))

print("Kendall")
print(normalised_kendall_tau_distance(rank, real_rank)*20*19)

print(Upsets(m, rank, unique, N, ldata)/msum)
startTime = time.time()
rank, upsets = ImproveRanking2(matrix, rank, unique, N, ldata)
print(time.time() - startTime)
print(upsets)
print(rank)

s1 = UpdateScore(last, unique, vec)
print(sum(Score(matrix, draws, s1, unique, N)))
"""
# Test
N = len(unique)
new_vec = [[0,1,1,1,1],[0,0,1,1,1],[0,0,0,1,1],[0,0,0,0,1],[0,0,0,0,0]]
new_vec = [[0,1,1,1,1,1],[0,0,1,1,1,1],[0,0,0,1,1,1],[0,0,0,0,1,1],[0,0,0,0,0,1],[0,0,0,0,0,0]]
new_vec = np.zeros([N,N])
for i in range(N):
    for j in range(N):
        if j > i:
            new_vec[i, j] = 2/N*N
u, s, vh = np.linalg.svd(new_vec)

print(s[0])
print(u[:,0])
print(s[1])
print(u[:,2])
print(StimesU(u, s, int(N/2), N))
"""

# SVD
print("SVD")
u, s, vh = np.linalg.svd(matrix)
i = 1
upsets = np.sum(m)
best = []
upsets_plt = np.zeros(N)
scores = np.zeros(N)
best_score = 0

while i < N:
    vec = np.absolute(StimesU(u, s, i, N))
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    rank = [tple[0] for tple in ranked]
    U = Upsets(m, rank, unique, N, ldata)
    print(i)
    
    upsets_plt[i] = U
    s1 = UpdateScore(last, unique, vec)
    scores[i] = sum(Score(matrix, draws, s1, unique, N))
    """
    if U < upsets:
        print(i)
        print(rank[0:10])
        print("Upsets")
        print(U)
        best = rank
        upsets = U
        v = vec
    """
    if scores[i] > best_score:
        print(i)
        print(rank[0:10])
        print("Upsets")
        print(U)
        print("score: " + str(scores[i]))
        best = rank
        best_score = scores[i]
        upsets = U
        v = vec
    i+=1

print("Kendall")
print(normalised_kendall_tau_distance(best, real_rank)*20*19)

plt.plot(upsets_plt)
plt.show()
plt.plot(scores)
plt.show()


print(best[0:10])
print("Upsets")
print(upsets)

print(upsets/msum)
startTime = time.time()
rank, upsets = ImproveRanking2(matrix, rank, unique, N, ldata)
print(time.time() - startTime)
print(upsets)
print(rank)

s1 = UpdateScore(last, unique, v)
print(sum(Score(matrix, draws, s1, unique, N)))


# Bradley-Terry

print("BT")
p = np.ones(N)
n = 0
d = 0
v = 0
upsets = np.sum(m)
scores = np.zeros(50)
upsets_plt = np.zeros(50)
best_score = 0
for t in range(50):
    for i in range(N):
        n = 0
        d = 0
        for j in range(N):
            if j != i:
                n += matrix[i][j]
                d += (matrix[i][j] + matrix[j][i])/(p[i]+p[j])

        p[i] = n/d
    p = p/np.linalg.norm(p)
    #print(p)

    vec = p
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    rank = [tple[0] for tple in ranked]
    U = Upsets(m, rank, unique, N, ldata)
    upsets_plt[t] = U
    print(t)
    s1 = UpdateScore(last, unique, vec)
    scores[t] = sum(Score(matrix, draws, s1, unique, N))
    print(scores[t])
    """
    if U < upsets:
        print(t)
        print(rank[0:10])
        print("Upsets")
        print(U)
        best = rank
        upsets = U
        v = vec
    """
    if scores[t] > best_score:
        print(t)
        print(rank[0:10])
        print("Upsets")
        print(U)
        print("score: " + str(scores[t]))
        best = rank
        best_score = scores[t]
        upsets = U
        v = vec

print("Kendall")
print(normalised_kendall_tau_distance(best, real_rank)*20*19)
"""
print(best[0:10])
print("Upsets")
print(upsets)
print(upsets/msum)
startTime = time.time()
rank, upsets = ImproveRanking2(matrix, rank, unique, N, ldata)
print(time.time() - startTime)
print(upsets)
print(rank)
"""
s1 = UpdateScore(last, unique, v)
print(sum(Score(matrix, draws, s1, unique, N)))

plt.plot(upsets_plt)
plt.show()
plt.plot(scores)
plt.show()

"""
print("model extraction")
file = "WC18.yml"
rank, r = ReadResults(file, unique, N)
rank = np.array(rank)
vec = rank[:,1].astype(np.int)
u = rank[:,0]
sorted_indices = vec.argsort()
ranked = [(u[i], vec[i]) for i in sorted_indices]
rank = [tple[0] for tple in ranked]
print(rank)
print(Upsets(m, rank, unique, N, d))
print(Upsets2(m, r, unique, N, d))


print("model extraction")
file = "results.yml"
rank, r = ReadResults2(file, unique, N)
rank = np.array(rank)
vec = rank[:,1]
u = rank[:,0]
sorted_indices = vec.argsort()
ranked = [(u[i], vec[i]) for i in sorted_indices]
rank = [tple[0] for tple in ranked]
rank = np.flip(rank)
print(rank)
print(Upsets(m, rank, unique, N, d))

print(len(unique))
print("model extraction")
file = "Testing.txt"
rank, r = ReadResults3(file, unique, N)
rank = np.array(rank)
vec = rank[:,1].astype(np.int)
u = rank[:,0]
sorted_indices = vec.argsort()
ranked = [(u[i], vec[i]) for i in sorted_indices]
rank = [tple[0] for tple in ranked]
print(rank)
print(len(rank))
print(Upsets(m, rank, unique, N, d))
print(Upsets2(m, r, unique, N, d))
"""