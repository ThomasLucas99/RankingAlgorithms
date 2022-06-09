from PythonFunctions import *

#file = "Data/Premier/2018_2019.csv"
file = "Data/FIFA/Foot.csv"
start = dt.datetime(1900, 1, 1)
date = dt.datetime(2018, 1, 1)
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

#read = np.array(ReadPremier(file))
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
"Norway", "Slovakia", "Romania", "Egypt", "Ecuador", "Republic of Ireland", "Qatar", "Costa Rica", "Cameroon", "Saudi Arabia",
"Ghana", "Mali", "Northern Ireland", "Greece", "Ivory Coast", "Jamaica", "Finland", "Venezuela", "Burkina Faso", "Bosnia and Herzegovina",
"Iceland", "Panama", "DR Congo", "Slovenia", "Albania", "North Macedonia", "South Africa", "United Arab Emirates", "El Salvador", "Bulgaria",
"Montenegro", "Cape Verde", "China PR", "Iraq", "Honduras", "Bolivia", "Israel", "Oman", "Curacao", "Guinea",
"Uganda", "Benin", "Uzbekistan", "Georgia", "Syria", "Haiti", "Zambia", "Gabon", "Jordan", "Bahrain",
"Armenia", "Luxembourg", "Belarus", "Lebanon", "Kyrgyzstan", "Congo", "Vietnam", "Palestine", "Trinidad and Tobago", "Madagascar",
"Kenya", "Mauritania", "India", "Cyprus", "Guinea-Bissau", "Estonia", "Sierra Leone", "North Korea", "New Zealand", "Kosovo",
"Namibia", "Niger", "Equatorial Guinea", "Thailand", "Tajikistan", "Libya", "Mozambique", "Kazakhstan", "Azerbaijan", "Zimbabwe",
"Guatemala", "Faroe Islands", "Togo", "Sudan", "Angola", "Antigua and Barbuda", "Philippines", "Malawi", "Central African Republic", "Tanzania",
"Comoros", "Turkmenistan", "Latvia", "Rwanda", "Lithuania", "Ethiopia", "Saint Kitts and Nevis", "Suriname", "Burundi", "Solomon Islands",
"Kuwait", "Nicaragua", "Liberia", "Lesotho", "Eswatini", "Hong Kong", "Botswana", "Afghanistan", "Gambia", "Yemen",
"Myanmar", "New Caledonia", "Malaysia", "Andorra", "Dominican Republic", "Maldives", "Taiwan", "Tahiti", "Singapore", "Fiji",
"Barbados", "Vanuatu", "Indonesia", "Papua New Guinea", "Bermuda", "South Sudan", "Grenada", "Nepal", "Belize", "Cambodia",
"Mauritius", "Puerto Rico", "Saint Vincent and the Grenadines", "Malta", "Guyana", "Saint Lucia", "Montserrat", "Cuba", "Chad", "Moldova",
"Macau", "Dominica", "Mongolia", "Bhutan", "Bangladesh", "Laos", "Brunei", "Sao Tome and Principe", "American Samoa", "Liechtenstein",
"Djibouti", "Samoa", "Somalia", "Cayman Islands", "Timor-Leste", "Seychelles", "Pakistan", "Tonga", "Aruba", "Bahamas",
"Eritrea", "Gibraltar", "Sri Lanka", "Turks and Caicos Islands", "Guam", "United States Virgin Islands", "British Virgin Islands", "San Marino"]

data = read[:, 1:6]
d = len(data)

data, d = KeepOnlyGames(data, real_rank, d)
print(d)

unique  = Countries(data)
N = len(unique)
print(N)

imp1 = 1
T = 99
Squares = np.zeros([N,T,T])
x = []
y = []
while imp1 <= T:
    imp0 = 1
    #print(str(((imp1-1)*T+(imp0-1))/(T*T)*100) + "%")
    while imp0 <= T:
        print(str(((imp1-1)*T+(imp0-1))/(T*T)*100) + "%")
        importance = [imp0, imp1]
        matrix, draws = np.array(MatchTableDrawsImportance(data, unique, N, importance))

        """
        #RELO
        rank = np.ones(N)*1000
        rank_score = rank.copy()
        t = 2000
        i = 0
        while rank_score[0] < 1850 and rank_score[-1] > 100 and i < t:
            rank = ELO(matrix, rank, unique, N, 1)
            vec = rank
            sorted_indices = vec.argsort()
            ranked = [(unique[i], vec[i]) for i in sorted_indices]
            ranked.reverse()
            rank_team = [tple[0] for tple in ranked]
            rank_score = [tple[1] for tple in ranked]
            i+=1
        rank_team = np.array(rank_team)
        for k in range(N):
            Squares[k, imp0-1, imp1-1] = np.where(rank_team == unique[k])[0][0]+1
        
        # Colley
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
        for k in range(N):
            Squares[k, imp0-1, imp1-1] = np.where(rank == unique[k])[0][0]+1
        
        
        # Eigen values
        w, v = np.linalg.eig(matrix)
        vec = np.absolute(v[:, np.argmax(w)])
        sorted_indices = vec.argsort()
        ranked = [(unique[i], vec[i]) for i in sorted_indices]
        ranked.reverse()
        rank = [tple[0] for tple in ranked]
        rank = np.array(rank)
        for k in range(N):
            Squares[k, imp0-1, imp1-1] = np.where(rank == unique[k])[0][0]+1
        
        
        # SVD
        u, s, vh = np.linalg.svd(matrix)
        i = 0
        upsets = np.sum(matrix)
        best = []

        while i < 5:
            vec = np.absolute(StimesU(u, s, i, N))
            sorted_indices = vec.argsort()
            ranked = [(unique[i], vec[i]) for i in sorted_indices]
            ranked.reverse()
            rank = [tple[0] for tple in ranked]
            U = Upsets(matrix, rank, unique, N, d)
            if U < upsets:
                best = rank
                upsets = U
                v = vec
            i+=1
        best = np.array(best)
        for k in range(N):
            Squares[k, imp0-1, imp1-1] = np.where(best == unique[k])[0][0]+1
        
        """
        # Bradley-Terry
        p = np.ones(N)
        n = 0
        d = 0
        v = 0
        t = 10
        upsets = 10000
        for s in range(t):
            for i in range(N):
                n = 0
                d = 0
                for j in range(N):
                    if j != i:
                        n += matrix[i][j]
                        d += (matrix[i][j] + matrix[j][i])/(p[i]+p[j])

                p[i] = n/d
            p = p/np.linalg.norm(p)
        vec = p
        sorted_indices = vec.argsort()
        ranked = [(unique[i], vec[i]) for i in sorted_indices]
        ranked.reverse()
        rank = [tple[0] for tple in ranked]
        rank = np.array(rank)
        for k in range(N):
            Squares[k, imp0-1, imp1-1] = np.where(rank == unique[k])[0][0]+1
        
        x.append(imp0)
        y.append(imp1)
        imp0+=1
    imp1+=1
for i in range(N):
    c = Flat(Squares[i])
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=c, cmap="viridis")
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower right", title="Ranking")
    ax.add_artist(legend1)
    plt.axis([0, T*1.25, 0, T+1])
    plt.title(unique[i])
    #plt.show()
    plt.savefig("BT/" + str(unique[i]) + '.png')
    plt.close(fig)
