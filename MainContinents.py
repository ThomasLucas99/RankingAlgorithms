from PythonFunctions import *
import pyomo.environ as pyo

def Continents(UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC, data):
    matrix = [[0]*6 for i in range(6)]
    d = len(data)
    c = [-1, -1]
    for i in range(d):
        if data[i][0] in UEFA:
            c[0] = 0
        elif data[i][0] in CONMEBOL:
            c[0] = 1
        elif data[i][0] in CONCACAF:
            c[0] = 2
        elif data[i][0] in CAF:
            c[0] = 3
        elif data[i][0] in AFC:
            c[0] = 4
        elif data[i][0] in OFC:
            c[0] = 5

        if data[i][1] in UEFA:
            c[1] = 0
        elif data[i][1] in CONMEBOL:
            c[1] = 1
        elif data[i][1] in CONCACAF:
            c[1] = 2
        elif data[i][1] in CAF:
            c[1] = 3
        elif data[i][1] in AFC:
            c[1] = 4
        elif data[i][1] in OFC:
            c[1] = 5

        matrix[c[0]][c[1]] += 1
    return matrix

def Merge(cont1, cont2, unique, matrix):
    print("merge")
    print(cont1)
    print(cont2)
    cont1 = np.array(cont1)
    cont2 = np.array(cont2)
    merged = cont1.copy()
    merged = merged.tolist()
    print(merged)
    N = len(cont1) + len(cont2)
    score = np.zeros(len(cont1))
    print(len(cont1))
    place = 0
    for i in range(len(cont2)):
        insert = np.where(unique == cont2[i])[0][0]

        j = 0
        score = np.zeros(len(cont1))
        while j < len(cont1):

            a = np.where(unique == cont1[j])[0][0]

            score[j] = score[j-1] + matrix[insert][a] - matrix[a][insert]
            j += 1
        print(cont2[i])
        new_score = score[place:]
        print(new_score)
        m = min(new_score)
        places = np.where(new_score == m)[0]
        print(places)
        for k in range(len(places)):
            print(places[k])
            place = places[k] + place
            insplace = place + i
            print(place)
            print(merged)
            merged.insert(insplace, cont2[i])
            break
        print(merged)
    return merged
            
def OptiMerge(continents):
    rank = continents[0]

    model = pyo.ConcreteModel()
    model.I = 
    model.J = 

    return rank

#file = "Data/Premier/2018_2019.csv"
file = "Data/FIFA/Foot.csv"
start = dt.datetime(1900, 1, 1)
date = dt.datetime(2014, 1, 1)
#date = dt.datetime(2020, 5, 27)
end = dt.datetime(2021, 1, 1)
#end = dt.datetime(2021, 5, 27)

read = np.array(ReadDataFromDate(file, date))
#read = np.array(ReadDataFromWC(file, date))
#read = np.array(ReadDataFromWCQ(file, date))
#read = np.array(ReadDataDates(file, date, end))
#read = np.array(ReadDataDatesWC(file, date, end))
#read = read[:22]
#read = read[:44]

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
"Puerto Rico", "Saint Vincent", "Guyana", "Saint Lucia", "Montserrat", "Cuba", "Dominica", "Cayman Islands", "Aruba", "Bahamas", "Turks and Caicos",
"US Virgin Islands", "British Virgin Islands", "Anguilla"]

CAF = ["Senegal",  "Morocco", "Algeria", "Tunisia", "Nigeria", "Egypt", "Cameroon", "Ghana", "Mali", "Ivory Coast",
"Burkina Faso", "DR Congo",  "South Africa", "Cape Verde", "Guinea", "Uganda", "Benin", "Zambia", "Gabon", "Congo",
"Madagascar", "Kenya", "Mauritania", "Guinea-Bissau", "Sierra Leone", "Namibia", "Niger", "Equa Guinea", "Libya", "Mozambique",
"Zimbabwe", "Togo", "Sudan", "Angola", "Malawi", "CAR", "Tanzania", "Comoros", "Rwanda", "Ethiopia",
"Burundi", "Liberia", "Lesotho", "Eswatini", "Botswana", "Gambia", "South Sudan", "Mauritius", "Chad", "Sao Tome",
"Djibouti", "Somalia", "Seychelles", "Eritrea"]

AFC = ["Iran", "Japan", "South Korea", "Australia", "Qatar", "Saudi Arabia", "UAE", "China PR", "Iraq", "Uzbekistan",
"Oman", "Syria", "Jordan", "Bahrain", "Lebanon", "Kyrgyzstan", "Vietnam", "Palestine", "India", "North Korea",
"Thailand", "Tajikistan", "Philippines", "Turkmenistan", "Kuwait", "Hong Kong", "Afghanistan", "Yemen", "Myanmar",  "Malaysia",
"Maldives", "Chinese Taipei", "Singapore", "Indonesia", "Nepal", "Cambodia", "Macau", "Mongolia", "Bhutan", "Bangladesh",
"Laos", "Brunei", "Timor-Leste", "Pakistan", "Sri Lanka", "Guam"]

OFC = ["New Zealand", "Solomon Islands", "New Caledonia", "Tahiti", "Fiji", "Vanuatu", "Papua New Guinea", "American Samoa", "Samoa", "Tonga"]

real = [OFC, AFC, CAF, UEFA, CONMEBOL, CONCACAF]
#real = UEFA
#real2 = CONMEBOL

data = read[:, 1:6]
d = len(data)
data, d = KeepOnlyGames(data, np.array(Flat(real)), d)
contmatrix = Continents(UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC, data)

csv = "MatrixTest.csv"
print(contmatrix)
WriteCSV(csv, contmatrix, ["OFC", "AFC", "CAF", "UEFA", "CONMEBOL", "CONCACAF"], 6)
maxim = 0
for i in range(6):
    for j in range(6):
        if i != j:
            ratio = contmatrix[i][j]/sumlc(np.array(contmatrix), i, i)
            if ratio > maxim:
                maxim = ratio
                x = i
                y = j

print(str(x) + " " + str(y) + " " + str(maxim))
r = len(real) 
c = np.zeros(r)
cdata = []
#print(data)
unique = []
N = []
m = []
rank = []


for i in range(r):
    cd, c[i] = KeepOnlyGames(data, real[i], d)
    cdata.append(cd)
    unique.append(np.array(real[i]))
    N.append(len(unique[i]))
    mat, draws = np.array(MatchTableDraws(cdata[i], unique[i], N[i]))
    m.append(mat)
"""
for i in range(r):
    w, v = np.linalg.eig(m[i])
    vec = np.absolute(v[:, np.argmax(w)])
    sorted_indices = vec.argsort()

    ranked = [(unique[i][j], vec[j]) for j in sorted_indices]
    ranked.reverse()
    rank.append([tple[0] for tple in ranked])
    print(rank[i])

"""
for i in range(r):
    rank_score = np.ones(N[i])*1000
    T = 2000
    j = 0
    print(i)
    while rank_score[0] < 1850 and rank_score[-1] > 100 and j < T:
        rank_score = ELO(m[i], rank_score, unique[i], N[i], 1)
        vec = rank_score
        sorted_indices = vec.argsort()
        ranked = [(unique[i][k], vec[k]) for k in sorted_indices]
        ranked.reverse()
        rank_team = [tple[0] for tple in ranked]
        j+=1
    rank.append(rank_team)
    print(rank[i])

merged = rank[0].copy()
new_unique = unique[0].copy().tolist()
for i in range(r-1):
    new_unique += unique[i+1].tolist()
    new_data, new_d = KeepOnlyGames(data, new_unique, d)
    new_matrix, new_draws = np.array(MatchTableDraws(new_data, new_unique, len(new_unique)))

    merged = Merge(merged, rank[i+1], np.array(new_unique), new_matrix)

print(merged)
m, draws = np.array(MatchTableDraws(data, np.array(Flat(real)), len(Flat(real))))
print(Upsets(m, merged, np.array(Flat(real)), len(Flat(real)), d))
"""
#matrix = prev_m
# SVD
startTime = time.time()
print("SVD")
u, s, vh = np.linalg.svd(matrix)
i = 1
upsets = np.sum(m)
best = []
upsets_plt = np.zeros(N)
scores = np.zeros(N)

while i < N:
    vec = np.absolute(StimesU(u, s, i, N))
    sorted_indices = vec.argsort()
    ranked = [(unique[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    rank = [tple[0] for tple in ranked]
    U = Upsets(m, rank, unique, N, d)
    print(i)
    
    upsets_plt[i] = U
    #s1 = UpdateScore(last, unique, vec)
    #scores[i] = sum(Score(matrix, s1, unique, N))
    if U < upsets:
        print(i)
        print(rank[0:10])
        print("Upsets")
        print(U)
        best = rank
        upsets = U
        v = vec
    i+=1
print("time: " + str(time.time() - startTime))
plt.plot(upsets_plt)
#plt.plot(scores)
plt.show()


print(best[0:10])
print("Upsets")
print(upsets)

print(upsets/msum)
startTime = time.time()
rank, upsets = ImproveRanking2(matrix, rank, unique, N, d)
print(time.time() - startTime)
print(upsets)
print(rank)

s1 = UpdateScore(last, unique, v)
print(sum(Score(matrix, s1, unique, N)))


# Bradley-Terry
startTime = time.time()
print("BT")
p = np.ones(N)
n = 0
d = 0
v = 0
T = 200
upsets = np.sum(m)
scores = np.zeros(T)
upsets_plt = np.zeros(T)
for t in range(T):
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
    #print(t)
    #s1 = UpdateScore(last, unique, vec)
    #scores[t] = sum(Score(matrix, s1, unique, N))
    #print(scores[t])
    if U < upsets:
        print(t)
        print(rank[0:10])
        print("Upsets")
        print(U)
        best = rank
        upsets = U
        v = vec

print("time: " + str(time.time() - startTime))
print(best[0:10])

print("Upsets")
print(upsets)
print(upsets/msum)
startTime = time.time()
rank, upsets = ImproveRanking2(matrix, rank, unique, N, d)
print(time.time() - startTime)
print(upsets)
print(rank)

s1 = UpdateScore(last, unique, v)
print(sum(Score(matrix, s1, unique, N)))

plt.plot(upsets_plt)
#plt.plot(scores)
plt.show()

print(msum)
print(minimum)
"""
