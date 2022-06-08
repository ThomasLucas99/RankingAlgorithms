# Parser for extracting the correct information from the database
import time
import numpy as np
import datetime as dt
from scipy.linalg import solve
from scipy.sparse.linalg import eigs
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import random
from PythonFunctions import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def integral(x):
    return np.log(1 + np.exp(x))

def Gauss(x):
    return np.exp(-x*x/5)

def dGauss(x):
    return -2 * x * Gauss(x)/5

def ReadYear(file, year):
    
    f = open(file + str(year) + ".csv", "r", encoding="utf-8")
    line = f.readline()
    line = f.readline()
    line = line.split(",")
    # date, winner name, loser name
    read = [[line[5], line[10], line[18], line[4], line[25]]]
    
    while f:
        line = f.readline()
        line = line.split(",")
        if line == [""]:
            break
        read += [[line[5], line[10], line[18], line[4], line[25]]]
    f.close()
    return read


def ReadDataFromDate(file, year):
    i = year
    f = open(file + str(i) + ".csv", "r", encoding="utf-8")
    line = f.readline()
    line = f.readline()
    line = line.split(",")
    read = [[line[5], line[10], line[18], line[4], line[25]]]
    f.close()
    first = 0

    while i < 2022:
        f = open(file + str(i) + ".csv", "r", encoding="utf-8")
        line = f.readline()
        if first == 0:
            first = 1
            line = f.readline()
        while f:
            line = f.readline()
            line = line.split(",")
            if line == [""]:
                break
            read += [[line[5], line[10], line[18], line[4], line[25]]]
        f.close()
        i += 1
    return read

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

def Players(data):
    read = np.array(data)
    players = read[:, [1,2]]
    return np.unique(players)

def MatchTable(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    for i in range(len(data)):
        #print(data)
        matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 1
    return matrix

def MatchTableImportanceTennis(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    for i in range(len(data)):
        # grand slam
        if data[i][3] == "G":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 10
        # tour finals
        elif data[i][3] == "F":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 5
        # atp 1000
        elif data[i][3] == "M":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 3
        #atp 250/500
        elif data[i][3] == "A":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 2
        else:
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 1

        if data[i][4] == "F":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] *= 5
        elif data[i][4] == "SF":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] *= 3
        elif data[i][4] == "QF":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] *= 2
        else:
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] *= 1

    return matrix

def MatchTableImpTennis(data, unique, N, imp):
    matrix = [[0]*N for i in range(N)]
    for i in range(len(data)):
        # grand slam et Tour Finals
        if data[i][3] == "G" or data[i][3] == "F":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += imp[1]*imp[0]
        # ATP 1000 et 500
        elif data[i][3] == "M" or data[i][3] == "A":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += imp[1]
        else:
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 1

        if data[i][4] == "F":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] *= 5
        elif data[i][4] == "SF":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] *= 3
        elif data[i][4] == "QF":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] *= 2
        else:
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] *= 1

    return matrix


def MatchTableImpFinals(data, unique, N, imp):
    matrix = [[0]*N for i in range(N)]
    for i in range(len(data)):
        # grand slam
        if data[i][3] == "G":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 10
        # tour finals
        elif data[i][3] == "F":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 5
        # atp 1000
        elif data[i][3] == "M":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 3
        #atp 250/500
        elif data[i][3] == "A":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 2
        else:
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 1

        if data[i][4] == "F":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] *= imp[0]*imp[1]
        elif data[i][4] == "SF" or data[i][4] == "QF":
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] *= imp[1]
        else:
            matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] *= 1

    return matrix

def MatchTableWR(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    matches = [[0]*N for i in range(N)]
    for i in range(len(data)):
        w = np.where(unique == data[i][1])[0][0]
        l = np.where(unique == data[i][2])[0][0]
        matrix[w][l] = (matrix[w][l]* matches[w][l] +1)/ (matches[w][l] +1)
        matches[w][l] += 1
    return matrix

def MatchTableSmall(data, unique, N):
    matrix = [[0]*N for i in range(N)]
    d = len(data)
    for i in range(d):
        matrix[np.where(unique == data[i][1])[0][0]][np.where(unique == data[i][2])[0][0]] += 1/d
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
    s = np.sum(matrix)
    matrix = matrix/s
    return matrix

def StimesU(u, s, n, N):
    if n > len(s):
        return []
    vec = np.zeros(N)
    for i in range(n):
        vec += s[i] * u[:,i]
    return vec

def WriteCSV(file, matrix, unique, N):
    f = open(file, "w+", encoding="utf-8")
    f.write("Players,")
    for i in range(N):
        f.write(str(unique[i]) + ",")

    for i in range(N):
        f.write("\n" + str(unique[i]) + ",")
        for j in range(N):
            f.write(str(matrix[i, j]) + ",")

    f.close()
    

def sumlc(matrix, l, c):
    s = np.sum(matrix[l])
    s += np.sum(matrix[:, c])
    return s

def Upsets(matrix, rank, unique, N, d):
    upsets = 0
    for i in range(N):
        for j in range(N):
            if i >= j:
                #print(np.where(unique == rank[i])[0])
                upsets += matrix[np.where(unique == rank[i])[0][0]][np.where(unique == rank[j])[0][0]]

    return upsets

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

def SetScore(last, matrix, unique, score, N, d):
    """
    for i in range(len(last)):
        for j in range(len(last[i])):
            #score[np.where(unique == last[i][j])] = i*2.5+2.5
            score[np.where(unique == last[i][j])] = i+1
            """

    for i in range(len(unique)):
        if score[i] == 0:
            score[i] = sum(matrix[i])*N/d
            #score[i] = random.randint(0, 10)
            #print(score[i])
    return score

def NoWin(matrix, unique, N):
    i = 0
    countries = []
    for i in range(len(unique)):
        if np.sum(matrix[i]) == 0:
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

def Flat(l):
    flat = []
    for i in range(len(l)):
        for j in range(len(l[i])):
            flat.append(l[i][j])
    return flat
"""
def Score(matrix, score_rank, unique, N):
    score = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if matrix[i][j] > 0:
                score[i] += sigmoid(score_rank[i] - score_rank[j]) * matrix[i][j]
            if matrix[j][i] > 0:
                score[i] += sigmoid(score_rank[j] - score_rank[i]) * matrix[j][i]
            #if draws[i][j] + draws[j][i] > 0:
            #    score[i] += Gauss(score_rank[j] - score_rank[i]) * (draws[i][j] + draws[j][i])
            
    #print(score)
    return score

def Score_deriv(matrix, score_rank, unique, N):
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
            #if draws[i][j] + draws[j][i] > 0:
            #    score[i] += dGauss(score_rank[i] - score_rank[j]) * (draws[i][j] + draws[j][i])
                #print("3 :" + str(score))
    #print(score)
    return score
    """
def Score(matrix, score_rank, unique, N):
    score = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if matrix[i][j] > 0:
                score[i] += sigmoid(score_rank[i] - score_rank[j]) * matrix[i][j]
            if matrix[j][i] > 0:
                score[i] -= sigmoid(score_rank[i] - score_rank[j]) * matrix[j][i]
            #if draws[i][j] + draws[j][i] > 0:
            #    score[i] += Gauss(score_rank[j] - score_rank[i]) * (draws[i][j] + draws[j][i])
            
    #print(score)
    return score

def Score_deriv(matrix, score_rank, unique, N):
    score = np.zeros(N)
    for i in range(N):
        for j in range(N):
            #print(str(i) + " " + str(j))
            #print(score)
            #print(score_rank)
            if matrix[i][j] > 0:
                score[i] += derivative(score_rank[j] - score_rank[i]) * matrix[i][j]
                if False:
                    print("Test")
                    print(j)
                    print(score[i])
                #print(score_rank[i] - score_rank[j])
                #print("1 :" + str(score))
            if matrix[j][i] > 0:
                score[i] -= derivative(score_rank[i] - score_rank[j]) * matrix[j][i]
                #print("2 :" + str(score))
            #if draws[i][j] + draws[j][i] > 0:
            #    score[i] += dGauss(score_rank[i] - score_rank[j]) * (draws[i][j] + draws[j][i])
                #print("3 :" + str(score))
    #print(score)
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
    score = score*2/maximum
    
    return score

def Armijo(matrix, unique, score_rank, f, grad, new_f, d, step, rho, b1):
    N = len(unique)

    while condition1(f, new_f, grad, d, step, b1):
        #print("a: " + str(step))
        step = step * rho
        new_score_rank = UpdateScore(last, unique, score_rank + step * grad)
        new_f = sum(Score(matrix, new_score_rank, unique, N))
        
    return step, new_f

def  Wolfe(matrix, unique, score_rank, grad, new_grad, step, rho, b2):
    N = len(unique)
    while condition2(grad, new_grad, step, b2):
        #print("w: " + str(step))
        step = step/rho
        new_score_rank = UpdateScore(last, unique, score_rank + step * grad)
        new_grad = Score_deriv(matrix, new_score_rank, unique, N)

    return step, new_grad

def condition1(f, new_f, grad, d, step, b1):
    #print("One: " + str(new_phi < phi + b1 * step))
    return new_f <= f + b1 * np.dot(grad, d) and step * max(grad) > 1e-4 #np.linalg.norm(grad) # np.dot(grad, d) # sum(np.absolute(grad))

def condition2(grad, new_grad, step, b2):
    #print("Two: " + str(new_grad/grad < b2))
    #print(np.dot(new_grad, d))
    #print(np.dot(grad, d))

    return np.dot(new_grad, d)/np.dot(grad, d) <= b2 and step * max(grad) < 1

def conditions(f, grad, new_f, new_grad, d, step, b1, b2):
    return condition1(f, new_f, grad, d, step, b1) or condition2(grad, new_grad, step, b2)

def get_rank(vec, unique):
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    rank = [tple[0] for tple in ranked]
    return rank

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

def ImproveRanking(matrix, rank, unique, N, d):
    UpsetsByC, upsets = UpsetsByCountries(matrix, rank, unique, N, d)
    new_rank = rank.copy()
    i = 0
    progress = 0
    while i < N:
        if i > progress:
            progress = i
        
        if len(UpsetsByC[i]) > 0:
            new_rank.insert(UpsetsByC[i][-1], new_rank.pop(i))
            new_upsets = Upsets(matrix, new_rank, unique, N, d)
            if new_upsets < upsets:
                rank = new_rank.copy()
                upsets = new_upsets
                i = UpsetsByC[i][-1]
            else:
                new_rank = rank.copy()
        i += 1

    return rank, upsets

def UpsetsByCountries(matrix, r, unique, N, d):
    upsets = []
    upset = 0
    old = 0
    for i in range(N):
        upsets.append([])
        old = upset
        for j in range(N):
            if i > j:
                upornot = matrix[np.where(unique == r[i])[0][0]][np.where(unique == r[j])[0][0]]
                if upornot > 0:
                    upsets[i].append(j)
                    upset += upornot

        #upsets[i] = upset - old
    return upsets, upset

file = "Data/Tennis/tennis_atp-master/atp_matches_"
# wta
#file = "Data/Tennis/tennis_wta-master/wta_matches_"
formatDate = '%Y%m%d'
start = dt.datetime(2018, 1, 1)
# wta
#start = dt.datetime(1931, 1, 1)
date = dt.datetime(2021, 1, 1)
end = dt.datetime(2021, 1, 1)

print("Reading Data")
#read = np.array(ReadDataFromDate(file, start.year))
read =  np.array(ReadYear(file, date.year))

print("Getting player's names")
unique = Players(read)
N = len(unique)
lread = len(read)
print(N)


imp1 = 1
T = 100
Squares = np.zeros([N,T,T])
x = []
y = []

while imp1 <= T:
    imp0 = 1
    print(str(((imp1-1)*T+(imp0-1))/(T*T)*100) + "%")
    while imp0 <= T:
        importance = [imp0, imp1]

        #print("Creating the match table")
        matrix = np.array(MatchTableImpTennis(read, unique, N, importance))
        """
        #RELO
        rank = np.ones(N)*1000
        rank_score = rank.copy()
        T = 10000
        i = 0
        while rank_score[0] < 1850 and rank_score[-1] > 100 and i < T:
            print(i)
            rank = ELO(matrix, rank, unique, N, 1)
            vec = rank
            sorted_indices = vec.argsort()
            ranked = [(unique[i], vec[i]) for i in sorted_indices]
            ranked.reverse()
            rank_team = [tple[0] for tple in ranked]
            rank_score = [tple[1] for tple in ranked]
            i+=1

        gradient = True
        best_upsets = 100000
        step = 10
        rho = 0.5
        b1 = 0
        b2 = 0.9
        T = 100
        rank_score = []
        score_rank = []

        scores = np.zeros(T)
        upsets_plt = np.zeros(T)
        g_plt = np.zeros(T)
        ranking = []
        new_gradient = []

        matrix, m, unique, N, last = RemoveNoWin(matrix, m, unique, N)
        flat_last = Flat(last)
        read, lread, removed_games = RemoveGames(read, flat_last, lread)
        rating = []

        if gradient == True:
            print("Gradient descent")
            j = 0
            score_rank = np.zeros(N)
            # get the teams that do not have wins or draws
            new_matrix, new_m, new_unique, new_N, last = RemoveNoWin(matrix, m, unique, N)
            flat_last = Flat(last)
            # set the scores
            score_rank = SetScore(last, matrix, unique, score_rank, N, lread)
            score_rank = UpdateScore(last, unique, score_rank)
            step = 10/np.linalg.norm(Score_deriv(matrix, score_rank, unique, N))

            vec = score_rank
            sorted_indices = vec.argsort()
            ranked = [(unique[i], vec[i]) for i in sorted_indices]
            ranked.reverse()
            rank = [tple[0] for tple in ranked]
            print(rank[0:10])
            #while rank_score[0] < 10 and rank_score[-len(flat_last)-1] > len(last)+1:
            for t in range(T):
                #take random sample of matches
                #new_read = PickData(read, unique)
                new_read = read
                new_unique = Players(new_read)
                new_data = new_read
                new_N = len(new_unique)
                new_d = len(new_data)
                new_matrix = np.array(MatchTable(new_data, new_unique, new_N))
        
                csv = "TennisMatrix.csv"
                WriteCSV(csv, matrix, unique, N)

                # compute gradient
                new_gradient = Score_deriv(matrix, score_rank, unique, N)
                print("t: " + str(t))
                print("gradient: " + str(np.linalg.norm(new_gradient)))
                step = max(new_gradient) * 0.01

                # update the step
        
                f = sum(Score(matrix, score_rank, unique, N))
                print("Score: " + str(f))
                print("step: " + str(step))
                grad = new_gradient
                d = np.sign(grad)
                new_score_rank = UpdateScore(last, unique, score_rank + step * grad)
                new_f = sum(Score(matrix, new_score_rank, unique, N))
                new_grad = Score_deriv(matrix, new_score_rank, unique, N)
                n = 0

                test_grad = grad
                vec = score_rank
                sorted_indices = vec.argsort()
                ranked = [(unique[i], vec[i], test_grad[i]) for i in sorted_indices]
                ranked.reverse()
                rank = [tple[0] for tple in ranked]
                rank_score = [tple[1] for tple in ranked]
                rank_grad = [tple[2] for tple in ranked]
                #while conditions(f, grad, new_f, new_grad, d, step, b1, b2):
                    #step, new_f = Armijo(matrix, unique, score_rank, f, grad, new_f, d, step, rho, b1)
                step, new_grad = Wolfe(matrix, unique, score_rank, grad, new_grad, step, rho, b2)
                step, new_f = Armijo(matrix, unique, score_rank, f, grad, new_f, d, step, rho, b1)
        
                #apply the gradient
                score_rank += new_gradient*step
                # set back the last's scores
                score_rank = UpdateScore(last, unique, score_rank)
                vec = score_rank
                sorted_indices = vec.argsort()
                ranked = [(unique[i], vec[i]) for i in sorted_indices]
                ranked.reverse()
                rank = [tple[0] for tple in ranked]
                rank_score = [tple[1] for tple in ranked]
                upsets = Upsets(m, rank, unique, N, lread)

                upsets_plt[t] = upsets
                s1 = UpdateScore(last, unique, vec)
                scores[t] = sum(Score(matrix, s1, unique, N))
                g_plt[t] = np.linalg.norm(new_grad)

                if upsets < best_upsets:
                    best_upsets = upsets
                    ranking = rank
                    print("j: " + str(j))
                    print("upsets: " + str(upsets))
                    print(ranking[:10])
                    r_score = rank_score[:10]
                    print(r_score)
                j += 1
    
            print(rank[:N])
            print(new_gradient[:N])
            r_score = rank_score[:N]
            for i in range(N):
                r_score[i] = int(r_score[i]*50)
            print(r_score)
            upsets = Upsets(m, rank, unique, N, lread)
            print(upsets)

            plt.plot(upsets_plt)
            plt.plot(scores)
            plt.show()
            plt.plot(g_plt)
            plt.show()

        #rating = score_rank

        matrix, m, unique, N, last = RemoveNoWin(matrix, m, unique, N)
        flat_last = Flat(last)
        read, lread, removed_games = RemoveGames(read, flat_last, lread)
        """

        # Colley
        b = np.zeros([N])
        C = np.zeros([N, N])
        for i in range(N):
            for j in range(N):
                if i == j:
                    C[i][j] = 2 + np.sum(matrix[j]) + np.sum(matrix[:, i])
                else:
                    C[i][j] = -matrix[i, j] - matrix[j, i]
        for i in range(N):
            b[i]= 1 + (np.sum(matrix[i]) - np.sum(matrix[:, i]))/2

        s = solve(C,b)
        z = [x for _,x in sorted(zip(s,unique))]
        rank = np.flip(z)
        for k in range(N):
            Squares[k, imp0-1, imp1-1] = np.where(rank == unique[k])[0][0]+1

        """
        print("Eigen")
        # Eigen values
        w, v = np.linalg.eig(matrix)
        vec = np.absolute(v[:, np.argmax(w)])
        sorted_indices = vec.argsort()
        ranked = [(unique[i], vec[i]) for i in sorted_indices]
        ranked.reverse()
        rank = [tple[0] for tple in ranked]
        for k in range(N):
            Squares[k, imp0-1, imp1-1] = np.where(rank == unique[k])[0][0]+1

        #rating += vec

        # SVD
        print("SVD")
        u, s, vh = np.linalg.svd(matrix)
        i = 1
        upsets = np.sum(m)
        rank = []
        best = []
        v = []
        while i < N:
            vec = np.absolute(StimesU(u, s, i, N))
            sorted_indices = vec.argsort()
            ranked = [(unique[i], vec[i]) for i in sorted_indices]
            ranked.reverse()
            rank = [tple[0] for tple in ranked]
            U = Upsets(m, rank, unique, N, lread)
            if U < upsets:
                print(i)
                print(rank[0:10])
                print("Upsets")
                print(U)
                best = rank
                upsets = U
                v = vec
            i+=1

        print(best[0:10])
        print("Upsets")
        print(upsets)
        print(np.sum(m))

        #rating += v

        print("BT")
        p = np.ones(N)
        n = 0
        d = 0
        v = 0
        upsets = np.sum(m)
        scores = np.zeros(100)
        upsets_plt = np.zeros(100)
        for t in range(100):
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
            U = Upsets(m, rank, unique, N, d)
            upsets_plt[t] = U
            print(t)
            s1 = UpdateScore(last, unique, vec)
            scores[t] = sum(Score(matrix, s1, unique, N))
            print(scores[t])
            if U < upsets:
                print(t)
                print(rank[0:10])
                print("Upsets")
                print(U)
                best = rank
                upsets = U
                v = vec
        
        print(best[0:10])
        print("Upsets")
        print(upsets)
        print(upsets/msum)
        startTime = time.time()
        #rank, upsets = ImproveRanking(matrix, rank, unique, N, lread)
        print(time.time() - startTime)
        print(upsets)
        print(rank)

        s1 = UpdateScore(last, unique, v)
        print(sum(Score(matrix, s1, unique, N)))
        """
        x.append(imp0)
        y.append(imp1)
        imp0+=1
    imp1+=1

for i in range(N):
    c = Flat(Squares[i])
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=c, cmap="viridis")
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Ranking")
    ax.add_artist(legend1)
    plt.axis([-T/5, T+1, 0, T+1])
    plt.title(unique[i])
    #plt.show()
    plt.savefig("TennisColleyT/" + str(unique[i]) + '.png')
    plt.close(fig)
