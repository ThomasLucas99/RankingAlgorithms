# Parser for extracting the correct information from the database
from PythonFunctions import *

file = "Data/FIFA/Foot.csv"
start = dt.datetime(1900, 1, 1)
date = dt.datetime(2020, 1, 1)
#date = dt.datetime(2017, 5, 27)
end = dt.datetime(2021, 1, 1)
#end = dt.datetime(2021, 5, 27)

read = np.array(ReadDataFromDate(file, date))
#read = np.array(ReadDataFromWC(file, date))
#read = np.array(ReadDataFromWCQ(file, date))
#read = np.array(ReadDataDates(file, date, end))
#read = np.array(ReadDataDatesWC(file, date, end))
#read = read[:22]
#read = read[:44]

file = "Data/Tennis/tennis_atp-master/atp_matches_"
formatDate = '%Y%m%d'
date = dt.datetime(2020, 1, 1)
data =  np.array(ReadYear(file, date.year))
ldata = len(data)
print(data)

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

"""
data = read[:, 1:6]
ldata = len(data)
print(ldata)
data, d = KeepOnlyGames(data, real_rank, ldata)
"""
unique = Countries(data)
#unique =  np.array(["a", "b", "c", "d"])
N = len(unique)
print(N)
m, draws = np.array(MatchTableDraws(data, unique, N))
"""
m = np.array([[0,1,1,0],
 [0,0,1,1],
 [0,0,0,1],
 [1,0,0,0]])
draws = np.zeros(N)
"""
csv = "Matrix.csv"

WriteCSV(csv, m, unique, N)
"""
real_rank = np.array(real_rank)
i = 0
while i < len(real_rank):
    if not real_rank[i] in unique:
        real_rank = np.delete(real_rank, i)
        i -= 1
    i += 1
"""
msum = np.sum(m)
minim = Minimum(m, unique)
print("m: " + str(minim))

#matrix = np.array(MatchTable(data, unique, N)).astype(np.float64)
#matrix, draws = np.array(MatchTableDraws(data, unique, N))
matrix = m.copy()
"""
u_rand = 0
s_rand = 0
for i in range(1000):
    vec = RandomRanking(N)
    rank = get_rank(vec, unique)
    u_rand += Upsets(m, rank, unique, N, ldata)
    s_rand += sum(Score(matrix, vec, unique, N))
    print(i)

print("Random")
print(u_rand/1000)
print(s_rand/1000)
"""
"""
# Real ranking
r, unique, m, matrix, N, data, ldata = ExtractRankingFile("fifa_ranking.csv", unique, m, matrix, date, data, ldata)

matrix, m, unique, N, last = RemoveNoWin(matrix, m, unique, N)
#matrix, m, unique, N, first = RemoveOnlyWin(matrix, m, unique, N)
flat_last = Flat(last)
#flat_first = Flat(first)
data, ldata, removed_games = RemoveGames(data, flat_last, ldata)
#data, ldata, removed_games = RemoveGames(data, flat_first, ldata)
"""
gradient = True
best_upsets = 100000
step = 1
rho = 0.9
b1 = 0.2
b2 = 0.21
phi = 1
init_value = np.zeros(N)
T = 10

scores = np.zeros(T)
upsets_plt = np.zeros(T)
g_plt = np.zeros(T)

ranking = []
new_gradient = []
start = time.time()
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
    step = 100/np.linalg.norm(Score_deriv(matrix, draws, score_rank, unique, N))
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

    #while rank_score[0] < 10 and rank_score[-len(flat_last)-1] > len(last)+1:
    for t in range(T):
    #take random sample of matches
        
        #new_read = PickData(read, unique)
        new_read = read
        #print(new_read)
        new_data = data
        new_unique = Countries(data)
        #new_unique = np.array(["a", "b", "c", "d"])
        new_N = len(new_unique)
        new_d = len(new_data)
        #new_matrix, new_draws = np.array(MatchTableDraws(new_data, new_unique, new_N))
        new_matrix = m.copy()
        new_draws = np.zeros(new_N)

        # compute gradient
        #new_gradient = new_Score(new_matrix, new_draws, score_rank, new_unique, unique)
        new_gradient = new_Score_deriv(new_matrix, new_draws, score_rank, new_unique, new_unique)
        #print(score_rank)
        #print("gradient: ")
        #print(new_gradient)
        #gradient = Score(matrix, draws, score_rank, unique, N)
        #print(score_rank)
        # apply the gradient
        #print("vector")
        #print(np.absolute(new_gradient))
        #score_rank += (new_gradient/max(np.absolute(new_gradient)))/1000
        
        # update the step
        
        f = sum(new_Score(matrix, new_draws, score_rank, unique, new_unique))
        #print("score: " + str(f))
        grad = new_gradient
        d = np.sign(grad)
        new_score_rank = UpdateScore(last, unique, score_rank + step * grad)
        new_f = sum(new_Score(matrix, new_draws, new_score_rank, unique, new_unique))
        new_grad = new_Score_deriv(matrix, new_draws, new_score_rank, unique, new_unique)
        c = 0
        while conditions(f, grad, new_f, new_grad, d, step, b1, b2):
            step, new_grad = Wolfe(matrix, draws, unique, new_unique, last, score_rank, grad, new_grad, d, step, rho, b2)
            step, new_f = Armijo(matrix, draws, unique, new_unique, last, score_rank, f, grad, new_f, d, step, rho, b1)
            
            """
            new_read = PickData(read, unique)
            new_unique = Countries(new_read)
            new_data = new_read[:, 1:6]
            new_N = len(new_unique)
            new_d = len(new_data)
            new_matrix, new_draws = np.array(MatchTableDrawsImportance(new_data, new_unique, new_N))
        
            # compute gradient
            #new_gradient = new_Score(new_matrix, new_draws, score_rank, new_unique, unique)
            new_gradient = new_Score_deriv(new_matrix, new_draws, score_rank, unique, new_unique)
            #gradient = Score(matrix, draws, score_rank, unique, N)
            #print(score_rank)
            # apply the gradient
            #print("vector")
            #print(np.absolute(new_gradient))
            #score_rank += (new_gradient/max(np.absolute(new_gradient)))/1000
        
            # update the step
        
            f = sum(new_Score(matrix, new_draws, score_rank, unique, new_unique))
            grad = new_gradient
            d = np.sign(grad)
            new_score_rank = UpdateScore(last, unique, score_rank + step * grad)
            new_f = sum(new_Score(matrix, new_draws, new_score_rank, unique, new_unique))
            new_grad = new_Score_deriv(matrix, new_draws, new_score_rank, unique, new_unique)
            """
            if c > 1:
                break

        #print("step: " + str(step))
        score_rank += new_gradient*step
        #print(score_rank)
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
        upsets = Upsets(m, rank, unique, N, d)

        upsets_plt[t] = upsets
        s1 = UpdateScore(last, unique, vec)
        scores[t] = sum(Score(matrix, draws, s1, unique, N))
        g_plt[t] = np.linalg.norm(new_grad)

        if upsets < best_upsets:
            best_upsets = upsets
            ranking = rank
            print("j: " + str(j))
            #print("upsets: " + str(upsets))
            #print(ranking[:10])
            r_score = rank_score[:10]
            #for i in range(10):
            #    r_score[i] = int(r_score[i]*20)
            #print(r_score)
        #print(rank_score[0])
        #print(rank_score[-len(flat_last)-1])
        j += 1
    
    print("time: ")
    print(time.time() -start)
    print(rank[:N])
    print(new_gradient[:N])
    r_score = rank_score[:N]
    for i in range(N):
        r_score[i] = int(r_score[i]*20)
    print(r_score)
    upsets = Upsets(m, rank, unique, N, ldata)
    print(upsets)
    
    
    #s1 = UpdateScore(last, unique, vec)
    #print(sum(Score(matrix, s1, unique, N)))

    plt.plot(upsets_plt)
    plt.plot(scores)
    plt.show()
    plt.plot(g_plt)
    plt.show()
    
else:
    matrix, m, unique, N, last = RemoveNoWin(matrix, m, unique, N)
    flat_last = Flat(last)
    data, ldata, removed_games = RemoveGames(data, flat_last, ldata)


#matrix = matrix + np.diag([1 for i in range(N)])
#matrix = MatrixNormalizationLine(matrix, N)
#matrix = MatrixNormalizationCol(matrix, N)
#matrix = MatrixNormalization(matrix, N)

print(msum)

#new_data = MakeDataGreat(data, unique)

csv = "Matrix.csv"

WriteCSV(csv, matrix, unique, N)
"""
# Real ranking
r, unique, m, matrix, N = ExtractRankingFile("fifa_ranking.csv", unique, m, matrix, date)

print(r[0:10])
print("Upsets")
print(Upsets(m, r, unique, N, ldata))

# Colley
new_rank = []
length = N
colley_matrix = matrix.copy()
colley_unique = unique.copy()
colley_m = m.copy()
colley_data = data.copy()
nmb_games = ldata
algo = 0
for n in range(N):
    print("Colley") 
    b = np.zeros([length])
    C = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            if i == j:
                C[i][j] = 2 + sumlc(colley_matrix, i, j)
            else:
                C[i][j] = -colley_matrix[i][j] - colley_matrix[j][i]
    for i in range(length):
        b[i]= 1 + (np.sum(colley_matrix[i]) - np.sum(colley_matrix[:, i]))/2
    s = solve(C,b)
    z = [x for _,x in sorted(zip(s,colley_unique))]
    rank = np.flip(z)
    upsets = Upsets(colley_m, rank, colley_unique, length, nmb_games)
    print("upsets computed")
    rank, upsets = ImproveRanking2(colley_matrix, rank.tolist(), colley_unique, length, nmb_games)
    print("rank improved")
    mr = MatrixRank(colley_data, np.array(rank), length)
    csv = "MatrixRank.csv"

    WriteCSV(csv, mr, rank, length)

    mr = np.array(mr)

    new_rank.append(rank[algo])
    length -= 1
    index = np.where(colley_unique == rank[algo])[0][0]
    colley_matrix = np.delete(colley_matrix, index, axis = 0)
    colley_matrix = np.delete(colley_matrix, index, axis = 1)
    colley_m = np.delete(colley_m, index, axis = 0)
    colley_m = np.delete(colley_m, index, axis = 1)
    colley_unique = np.delete(colley_unique, index)
    colley_data, nmb_games, removed_games = RemoveGames(colley_data, rank[algo], nmb_games)
    rank.pop(0)

    print(rank)
    print(new_rank)
    print(upsets)

print("Upsets")
new_rank = np.array(new_rank)
if algo == -1:
    new_rank = np.flip(new_rank)

upsets = Upsets(m, new_rank, unique, N, ldata)
print(upsets)
print(upsets/msum)

startTime = time.time()
new_rank, upsets = ImproveRanking2(matrix, new_rank.tolist(), unique, N, ldata)
print(time.time() - startTime)
print(upsets)
print(new_rank)
print(Score(matrix, score_rank, unique, N))


# Eigen values
print("eigens")
from scipy.sparse.linalg import eigs
m = dok_matrix(m)
m = m.asfptype()
val,vec = eigs(m, which='LM', k=1)
vec = np.ndarray.flatten(abs(vec))
sorted_indices = vec.argsort()
ranked = [(unique[i], vec[i]) for i in sorted_indices]
ranked.reverse()
rank = [tple[0] for tple in ranked]
print(rank[0:10])
print(Upsets(matrix, rank, unique, N, ldata))
#rank, upsets = ImproveRanking2(matrix, rank, unique, N, ldata)
#print(upsets)


print("Eigen")
w, v = np.linalg.eig(matrix)
vec = np.absolute(v[:, np.argmax(w)])
sorted_indices = vec.argsort()
ranked = [(unique[i], vec[i]) for i in sorted_indices]
ranked.reverse()
rank = [tple[0] for tple in ranked]
print(rank[0:10])
print("Upsets")
print(Upsets(matrix, rank, unique, N, ldata))
startTime = time.time()
#rank, upsets = ImproveRanking2(matrix, rank, unique, N, ldata)
print(time.time() - startTime)
#print(upsets)
#print(rank)
score_rank = np.zeros(N)
for i in range(N):
    score_rank[i] = N - i
#print(Score(matrix, score_rank, unique, N, ldata))

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

# SVD
print("SVD")
u, s, vh = np.linalg.svd(matrix)
i = 1
upsets = np.sum(m)
best = []
while i < N:
    vec = np.absolute(StimesU(u, s, i, N))
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    rank = [tple[0] for tple in ranked]
    U = Upsets(m, rank, unique, N, ldata)
    if U < upsets:
        print(i)
        print(rank[0:10])
        print("Upsets")
        print(U)
        best = rank
        upsets = U
    i+=1

print(best[0:10])
print("Upsets")
print(upsets)
print(upsets/msum)
startTime = time.time()
rank, upsets = ImproveRanking(matrix, rank, unique, N, ldata)
print(time.time() - startTime)
print(upsets)
print(rank)

# PageRank Testing

print("PageRank")

class Node(object):
    def __init__(self, name):
        self.name = name
        self.losses = {} #Key = other team, value = sum margin of losses (point differential)

    def add_loss(self, oTeam, pointDiff):
        self.losses[oTeam] = self.losses.get(oTeam, 0) + pointDiff
        #Either adds a new loss, or updates the overall pointDiff

    def __str__(self):
        return str(self.losses)

def build_graph(games):
    '''Processes all games and create each team as a node object
    INPUT: games - array of games in format games[i] = [loser, winner, pointDiff]
    OUTPUT: nodes - with nodes['team name'] = object node of team '''
    nodes = {} #dictionary Team Name : Team Object
    for item in games: 
        loser = item[0]
        winner = item[1]
        pointDiff = item[2]
        nodes[loser]= nodes.get(loser, Node(loser))
        nodes[loser].add_loss(winner, pointDiff)
    return nodes

def build_matrix(nodes):
    '''Builds the point differential matrix from nodes
    INPUT:
        nodes - with nodes['team name'] = object node of team 
    OUTPUT: 
        team_index - team_index['team name'] = index corresponding to row & column in A
        A - point differential matrix with A[row][column]
            Rows are losers, and entries are the point differential between column team
    '''
    A = [[0 for x in range(len(nodes.keys()))] for x in range(len(nodes.keys()))] 
    teams = sorted(nodes.keys()) #array of teams alphabetically
    team_index = {}

    for i in range(len(teams)): 
        team_index[teams[i]] = i

    for i in range(len(teams)):
        node = nodes[teams[i]]
        for item in node.losses.keys(): 
            pointDiff = node.losses[item]
            col_index = team_index[item]
            A[i][col_index] = float(pointDiff)
    return A, team_index

def markovMatrix(matrixA):
    '''Creates Markov matrix by dividing each entry in A by the sum of the row
    INPUT: A - point differential matrix with A[row][column]
            Rows are losers, and entries are the point differential between col team
    OUTPUT: H - Markov matrix
    '''
    A = np.array(matrixA)
    H = [[0 for x in range(len(A))] for x in range(len(A))] 
    for i in range(len(A)):
        row_sum = np.sum(A[i])
        for j in range(len(A[i])):
            H[i][j] = float(A[i][j])/row_sum
    return H

def pageRank(matrixA):
    '''Returns the left eigenvector (the PageRank vector) of the input matrix
    INPUT: H - Markov matrix
    OUTPUT: vl - left eigenvector, or the PageRank vector
    '''
    A = np.array(matrixA)
    H = markovMatrix(A)
    w, vl, vr = linalg.eig(H, left = True)
    vl = np.absolute(vl[:,0].T)
    return vl

def printResults(vl, team_index):
    '''Uses the team_index dictionary to interpret the results from the PageRank vector
    INPUT: 
        vl - left eigenvector, or the PageRank vector
        team_index - team_index['team name'] = index corresponding to row & column in A
    OUTPUT: 
        top10: arrray of top10 teams based on PageRank vector
    '''
    top10= []
    teams = sorted(team_index.keys())
    for i in range(N):
        ind = np.argmax(vl)
        top10.append(teams[ind])
        vl[ind] = 0
    return top10

nodes = build_graph(new_data)
A, team_index =  build_matrix(nodes)
pi = pageRank(A)
#pprint.pprint(pi)
rank = printResults(pi, team_index)
pprint.pprint(rank)

print("Upsets")
print(Upsets(m, rank, unique, N, ldata))
print(Upsets(m, rank, unique, N, ldata)/msum)


def Upsets2(matrix, r, unique, N, ldata):
    upsets = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    if j > l:
                        upsets += matrix[i][k] * r[i,j] * r[k,l]
    print("Upsets")
    return upsets

print("model extraction")
file = "results.yml"
rank, r = ReadResults(file, unique, N)
rank = np.array(rank)
vec = rank[:,1].astype(np.int)
u = rank[:,0]
sorted_indices = vec.argsort()
ranked = [(u[i], vec[i]) for i in sorted_indices]
rank = [tple[0] for tple in ranked]
print(rank)
print(Upsets(m, rank, unique, N, ldata))
print(Upsets2(m, r, unique, N, ldata))


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
print(Upsets(m, rank, unique, N, ldata))


def Score(matrix, unique, N, ldata):
    score = np.zeros(N)
    for i in range(N):
        score[i] = np.sum(matrix[i,:])
        score[i] -= np.sum(matrix[:,i])
    return score

nmb_end = 0
s = []
s.append(unique.tolist())
rank = ['' for i in range(N)]
cntr = 0
print(N)
while cntr < 20:
    new_s = []
    sl = 0
    length_s = len(s)
    #print(length_s)
    #flat_s = Flat(s)
    #print(len(flat_s))
    #print(nmb_end)
    for n in range(length_s):
        new_N = len(s[n])
        sl += new_N
        #print(sl)
        if new_N >= 1:
            new_data, new_d = KeepOnlyGames(data, s[n], ldata)
            
            new_matrix = np.array(MatchTable(new_data, np.array(s[n]), new_N))
            #print(new_data)
            score = Score(matrix, s[n], new_N, new_d)
            #print(score)
            for a in range(2):
                new_s.append([])
            #print(s)
            for i in range(len(score)):
                if score[i] > 0:
                    new_s[n].append(s[n][i])
                else:
                    new_s[1+n].append(s[n][i])
            #print(len(new_s[0]))
            #print(len(new_s[1]))
               
        elif new_N == 1:
            sl -= 1
            new_sl = sl
            
            while rank[new_sl] != '':
                new_sl += 1
                #print(rank[sl])
                #print(s[n][0])
            rank[new_sl] = s[n][0]
            
            nmb_end += 1
            
            new_s.append([])
            new_s.append([])
            new_s[n+1].append(s[n][0])
        else:
            for a in range(2):
                new_s.append([])
        #print(new_s)
    print(Flat(s))
    print(rank)
    s = new_s
    cntr += 1

#print(s)
flat_s = Flat(s)
print(flat_s)
print(len(flat_s))
print(len(rank))
print(N)
j = 0
for i in range(len(rank)):
    if rank[i] == '':
        rank[i] = flat_s[j]
        j+=1
print(rank)

print(Upsets(m, rank, unique, N, ldata))
#rank, upsets = ImproveRanking2(matrix, rank, unique, N, ldata)
#print(upsets)
"""
