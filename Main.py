from PythonFunctions import *

def atp(filename):
    f = open(filename, "r")
    count = 0
    rank = []
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = line.replace("\n", "")
    rank = [line]
    line = f.readline()
    line = f.readline()
    while f :
        line = f.readline()
        line = line.replace("\n", "")
        if line == "":
            break
        if line[0] == "-" or line[0] == "+":
            count -= 1
        if count == 2:
            rank += [line]
        if count == 4:  
            count = -1
        count+=1
    return rank

#file = "Data/Premier/2017_2018.csv"
file = "Data/FIFA/Foot.csv"
start = dt.datetime(1900, 1, 1)
date = dt.datetime(2018, 1, 1)
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

#read = np.array(ReadPremier(file))

file = "Data/Tennis/tennis_atp-master/atp_matches_"
formatDate = '%Y%m%d'
date = dt.datetime(2020, 1, 1)
data =  np.array(ReadYear(file, date.year))
d = len(data)
print(data)

rank = np.array(atp("ATPRanking.txt"))
print(rank)


"""
# 18-19
real_rank = ['Man City', 'Liverpool', 'Chelsea', 'Tottenham', 'Arsenal', 'Man United', 'Wolves', 'Everton', 'Leicester', 'West Ham', 'Watford', 'Crystal Palace', 'Newcastle', 'Bournemouth', 'Burnley', 'Southampton', 'Brighton', 'Cardiff', 'Fulham', 'Huddersfield']
real_score = [98, 97, 72, 71, 70, 66, 57, 54, 52, 52, 50, 49, 45, 45, 40, 39, 36, 34, 26, 16]
real_unique = ['Arsenal', 'Bournemouth', 'Brighton', 'Burnley', 'Cardiff', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Huddersfield', 'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Southampton', 'Tottenham', 'Watford', 'West Ham', 'Wolves']
real_score = [70, 45, 36, 40, 34, 72, 49, 54, 26, 16, 52, 97, 98, 66, 45, 39, 71, 50, 52, 57]


# 17-18
real_rank = ['Man City', 'Man United', 'Tottenham', 'Liverpool', 'Chelsea', 'Arsenal', 'Burnley', 'Everton', 'Leicester', 'Newcastle', 'Crystal Palace', 'Bournemouth', 'West Ham', 'Watford', 'Brighton', 'Huddersfield', 'Southampton', 'Swansea', 'Stoke', 'West Brom']
real_score = [100, 81, 77, 75, 70, 63, 54, 49, 47, 44, 44, 44, 42, 41, 40, 37, 36, 33, 33, 31]
real_unique = ['Arsenal', 'Bournemouth', 'Brighton', 'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Huddersfield', 'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 'Southampton', 'Stoke', 'Swansea', 'Tottenham', 'Watford', 'West Brom', 'West Ham']
real_score = [63, 44, 40, 54, 70, 44, 49, 37, 47, 75, 100, 81, 44, 36, 33, 33, 77, 41, 31, 42]

"""
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

UEFA = ["Belgium", "France", "England", "Italy", "Spain", "Portugal", "Denmark", "Netherlands", "Germany", "Switzerland",
"Croatia", "Sweden", "Wales", "Serbia", "Ukraine", "Poland", "Austria", "Czech Republic", "Russia", "Turkey",
"Scotland", "Hungary", "Norway", "Slovakia", "Romania", "Ireland", "Nothern Ireland", "Greece", "Finland", "Bosnia and Herzegovina",
"Iceland", "Slovenia", "Albania", "North Macedonia", "Bulgaria", "Montenegro", "Israel", "Georgia", "Armenia", "Luxembourg",
"Belarus", "Cyprus", "Estonia", "Kosovo", "Kazakhstan", "Azerbaijan", "Faroe Islands",  "Latvia", "Lithuania", "Andorra",
"Malta", "Moldova", "Liechtenstein", "Gibraltar", "San Marino"]

CONMEBOL = ["Brazil", "Argentina", "Colombia", "Uruguay",  "Peru",  "Chile", "Paraguay", "Ecuador", "Venezuela", "Bolivia"]

CONCACAF = ["United States", "Mexico", "Canada", "Costa Rica", "Jamaica", "Panama", "El Salvador", "Honduras", "Curacao", "Haiti", "Trinidad and Tobago",
"Guatemala", "Antigua and Barbuda", "Saint Kitts and Nevis", "Suriname", "Nicaragua", "Dominican Republic", "Barbados", "Bermuda", "Grenada", "Belize",
"Puerto Rico", "Saint Vincent and the Grenadines", "Guyana", "Saint Lucia", "Montserrat", "Cuba", "Dominica", "Cayman Islands", "Aruba", "Bahamas", "Turks and Caicos Islands",
"United States Virgin Islands", "British Virgin Islands", "Anguilla"]

CAF = ["Senegal",  "Morocco", "Algeria", "Tunisia", "Nigeria", "Egypt", "Cameroon", "Ghana", "Mali", "Ivory Coast",
"Burkina Faso", "DR Congo",  "South Africa", "Cape Verde", "Guinea", "Uganda", "Benin", "Zambia", "Gabon",
"Congo", "Madagascar", "Kenya", "Mauritania", "Guinea-Bissau", "Sierra Leone", "Namibia", "Niger", "Equatorial Guinea", "Libya",
"Mozambique", "Zimbabwe", "Togo", "Sudan", "Angola", "Malawi", "Central African Republic", "Tanzania", "Comoros", "Rwanda",
"Ethiopia", "Burundi", "Liberia", "Lesotho", "Eswatini", "Botswana", "Gambia", "South Sudan", "Mauritius", "Chad", "Sao Tome and Principe", "Djibouti", "Somalia", "Seychelles", "Eritrea"]

AFC = ["Iran", "Japan", "South Korea", "Australia", "Qatar", "Saudi Arabia", "United Arab Emirates", "China PR", "Iraq", "Uzbekistan",
"Oman", "Syria", "Jordan", "Bahrain", "Lebanon", "Kyrgyzstan", "Vietnam", "Palestine", "India", "North Korea",
"Thailand", "Tajikistan", "Philippines", "Turkmenistan", "Kuwait", "Hong Kong", "Afghanistan", "Yemen", "Myanmar",  "Malaysia",
"Maldives", "Chinese Taipei", "Singapore", "Indonesia", "Nepal", "Cambodia", "Macau", "Mongolia", "Bhutan", "Bangladesh",
"Laos", "Brunei", "Timor-Leste", "Pakistan", "Sri Lanka", "Guam"]

OFC = ["New Zealand", "Solomon Islands", "New Caledonia", "Tahiti", "Fiji", "Vanuatu", "Papua New Guinea", "American Samoa", "Samoa", "Tonga"]

"""
data = read[:, 1:6]
d = len(data)
print(data)
"""
data, d = KeepOnlyGames(data, rank, d)
unique = rank
#unique = np.array(["a", "b", "c", "d"])
N = len(unique)

m, draws = np.array(MatchTableDraws(data, unique, N))
upsets = Upsets(m, rank, unique, N, d)
print(upsets)
"""
m = np.array([[0,1,1,0],
 [0,0,1,1],
 [0,0,0,1],
 [1,0,0,0]])
"""
msum = np.sum(m)

minimum = Minimum(m, unique)

matrix = np.array(MatchTable(data, unique, N)).astype(np.float64)
#matrix = m.copy()

csv = "MatrixTest.csv"

WriteCSV(csv, matrix, unique, N)
print(N)
print(d)

#matrix = matrix + np.diag([1/d for i in range(N)])
#matrix = MatrixNormalizationLine(matrix, N)
#matrix = MatrixNormalizationCol(matrix, N)
#matrix = MatrixNormalization(matrix, N)

#print(msum)

#new_data = MakeDataGreat(data, unique)

# Real ranking
#r, unique, m, matrix, N, data, d = ExtractRankingFile("fifa_ranking.csv", unique, m, matrix, date, data, d)
"""
real_rank = np.array(real_rank)
i = 0
while i < len(real_rank):
    if not real_rank[i] in unique:
        real_rank = np.delete(real_rank, i)
        i -= 1
    i += 1

print("real")
print(Upsets(m, real_rank, np.array(unique), N, d))
"""
msum = sum(sum(m))
#matrix, m, unique, N = RemoveNoMatch(matrix, m, unique, N)

#matrix, m, unique, N, last = RemoveNoWin(matrix, m, unique, N)
#matrix, m, unique, N, first = RemoveOnlyWin(matrix, m, unique, N)
#flat_last = Flat(last)
#flat_first = Flat(first)
#data, d, removed_games = RemoveGames(data, flat_last, d)
#data, d, removed_games = RemoveGames(data, flat_first, d)

#print(r[0:10])
#print("Upsets")
#print(Upsets(m, r, unique, N, d))

#matrix = IndirectMatrix(matrix, unique, N)

csv = "Matrix.csv"

WriteCSV(csv, matrix, unique, N)

#ELO
#rank = np.ones(N)*1000
rank_score = np.ones(N)*1000
T = 100
i = 0
upsets_plt = np.zeros(T)
startTime = time.time()
while np.max(rank_score) < 2000 and np.min(rank_score) > 0 and i < T:
    rank_score = ELO(matrix, rank_score, unique, N, 2)
    #vec = rank_score
    #sorted_indices = vec.argsort()
    #ranked = [(unique[j], vec[j]) for j in sorted_indices]
    #ranked.reverse()
    #rank_team = [tple[0] for tple in ranked]
    #upsets = Upsets(m, rank_team, unique, N, d)
    #upsets_plt[i] = upsets
    i+=1
    #print(i)
print("time: " + str(time.time() - startTime))
#plt.plot(upsets_plt[0:i])
#plt.plot(scores)
#plt.show()

#print(rank_team[0:10])
#print(rank_score[0:10])
vec = rank_score
sorted_indices = vec.argsort()
ranked = [(unique[j], vec[j]) for j in sorted_indices]
ranked.reverse()
rank_team = [tple[0] for tple in ranked]
upsets = Upsets(m, rank_team, unique, N, d)
print("Upsets: " + str(upsets))

# Colley
print(N)
print("Colley")
b = np.zeros([N])
C = np.zeros([N, N])
startTime = time.time()
for i in range(N):
    for j in range(N):
        if i == j:
            C[i][j] = 2 + sumlc(matrix, i, j)
        else:
            C[i][j] = -matrix[i][j] - matrix[j][i]
for i in range(N):
    b[i]= 1 + (np.sum(matrix[i]) - np.sum(matrix[:, i]))/2
s = solve(C,b)
z = [x for _,x in sorted(zip(s,unique))]
rank = np.flip(z)
print("time: " + str(time.time() - startTime))
upsets = Upsets(m, rank, unique, N, d)
print(upsets)
"""
s1 = s
z = [x for _,x in sorted(zip(s,unique))]
rank = np.flip(z)
print(rank[0:10])
print("Upsets")
upsets = Upsets(m, rank, unique, N, d)
print(upsets)
print(upsets/msum)

startTime = time.time()
rank, upsets = ImproveRanking2(matrix, rank.tolist(), unique, N, d)
print(time.time() - startTime)
print(upsets)
print(rank)
s1 = UpdateScore(last, unique, s1)
#print(sum(Score(matrix, s1, unique, N)))
print(msum)
"""
"""
rank = np.array(rank)
mr = MatrixRank(data, rank, N)
csv = "MatrixRank.csv"

WriteCSV(csv, mr, rank, N)
"""

# Eigen values

csv = "MatrixRank.csv"
WriteCSV(csv, matrix, unique, N)

startTime = time.time()
print("Eigen")
w, v = np.linalg.eig(matrix)
print("time: " + str(time.time() - startTime))
vec = np.absolute(v[:, np.argmax(w)])
sorted_indices = vec.argsort()
ranked = [(unique[i], vec[i]) for i in sorted_indices]
ranked.reverse()
rank = [tple[0] for tple in ranked]
print("time: " + str(time.time() - startTime))
print(rank[0:10])
print("Upsets")
print(Upsets(m, rank, unique, N, d))

"""
print(Upsets(m, rank, unique, N, d)/msum)
startTime = time.time()
rank, upsets = ImproveRanking2(matrix, rank, unique, N, d)
print(time.time() - startTime)
print(upsets)
print(rank)

s1 = UpdateScore(last, unique, vec)
print(sum(Score(matrix, s1, unique, N)))
"""

#matrix = prev_m
# SVD
print("SVD")
i = 0
upsets = np.sum(m)
best = []
upsets_plt = np.zeros(2)
scores = np.zeros(N)
startTime = time.time()
u, s, vh = np.linalg.svd(matrix)

while i < 2:
    #vec = np.absolute(StimesU(u, s, i, N))
    vec = np.absolute(u[:,i])
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    rank = [tple[0] for tple in ranked]
    U = Upsets(m, rank, unique, N, d)
    
    #upsets_plt[i] = U
    #s1 = UpdateScore(last, unique, vec)
    #scores[i] = sum(Score(matrix, s1, unique, N))
    if U < upsets:
        #print(i)
        #print(rank[0:10])
        #print("Upsets")
        #print(U)
        best = rank
        upsets = U
        v = vec
    i+=1
print("time: " + str(time.time() - startTime))
#plt.plot(upsets_plt)
#plt.plot(scores)
#plt.show()


#print(best[0:10])
#print("Upsets")
print(upsets)

"""
print(upsets/msum)
startTime = time.time()
rank, upsets = ImproveRanking2(matrix, rank, unique, N, d)
print(time.time() - startTime)
print(upsets)
print(rank)

s1 = UpdateScore(last, unique, v)
print(sum(Score(matrix, s1, unique, N)))
"""

# Bradley-Terry
print("BT")
p = np.ones(N)
new_p = np.ones(N)
n = 0
d = 0
v = 0
T = 100
upsets = np.sum(m)
scores = np.zeros(T)
upsets_plt = np.zeros(T)
startTime = time.time()
for t in range(T):
    for i in range(N):
        n = 0
        d = 0
        for j in range(N):
            if j != i:
                n += matrix[i][j]
                if p[i]+p[j] != 0:
                    d += (matrix[i][j] + matrix[j][i])/(p[i]+p[j])
        new_p[i] = n/d
    p = new_p/np.linalg.norm(p)
    #vec = p
    #sorted_indices = vec.argsort()
    #ranked = [(unique[i], vec[i]) for i in sorted_indices]
    #ranked.reverse()
    #rank = [tple[0] for tple in ranked]
    #U = Upsets(m, rank, unique, N, d)
    #upsets_plt[t] = U
    #print(t)
    #s1 = UpdateScore(last, unique, vec)
    #scores[t] = sum(Score(matrix, s1, unique, N))
    #print(scores[t])
    #if U < upsets:
        #print(t)
        #print(rank[0:10])
        #print("Upsets")
        #print(U)
        #best = rank
        #upsets = U
        #v = vec

vec = p
sorted_indices = vec.argsort()
ranked = [(unique[i], vec[i]) for i in sorted_indices]
ranked.reverse()
rank = [tple[0] for tple in ranked]
U = Upsets(m, rank, unique, N, d)
print(U)
print("time: " + str(time.time() - startTime))
print(best[0:10])

plt.plot(upsets_plt)
#plt.plot(scores)
plt.show()

print(msum)
print(minimum)
