from PythonFunctions import *

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

#read = read[0:130]
t = 100
i = 0
data = read[:, 1:6]
d = len(data)
data, d = KeepOnlyGames(data, real_rank, d)

new_unique = []
unique = Countries(data)
old_unique = unique.copy()
N = len(unique)
UN = len(unique)
old_N = len(unique)
print(d)

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
        #matrix, draws = np.array(MatchTableDrawsImportance(data, np.array(unique), N, importance))
        unique = [old_unique.copy()]
        time = 0
        end = 0
        while end != len(unique) and end < UN:
            end = 0
            for j in range(len(unique)):
                N = len(unique[j])
                if N > 1:
                    new_data, new_d = KeepOnlyGames(data, np.array(unique[j]), d)

                    matrix, draws = np.array(MatchTableDrawsImportance(new_data, np.array(unique[j]), N, importance))
                    m = matrix.copy()

                    matrix, m, unique[j], N, last = RemoveNoWin(matrix, m, unique[j], N)
                    matrix, m, unique[j], N, first = RemoveOnlyWin(matrix, m, unique[j], N)
                    flat_last = Flat(last)
                    flat_first = Flat(first)
                    data, d, removed_games = RemoveGames(data, flat_last, d)
                    data, d, removed_games = RemoveGames(data, flat_first, d)
            
                    matrix = matrix.tolist()
                    if matrix != []:
                        model = pyo.ConcreteModel()

                        model.I = pyo.RangeSet(0, N-1)

                        model.r = pyo.Var(model.I, domain = pyo.NonNegativeIntegers, bounds=(0,1))

                        def obj_expression(model):
                            return Clusters(matrix, model.r, unique[j], N, new_d)

                        model.OBJ = pyo.Objective(rule=obj_expression)
                        """
                        model.constraints = pyo.ConstraintList()
                        model.constraints.add(expr = sum(model.r[i] for i in model.I) >= 1)
                        model.constraints.add(expr = sum(model.r[i] for i in model.I) <= N-1)
                        """
                        model.constraints = pyo.ConstraintList()
                        model.constraints.add(expr = sum(model.r[i] for i in model.I) >= int(N/2))
                        model.constraints.add(expr = sum(model.r[i] for i in model.I) <= int((N+1)/2))
                        original_stdout = sys.stdout
                        f = open('Warning.txt', 'w+')
                        sys.stdout = f
                        solver = pyo.SolverFactory('cplex')
                        results = solver.solve(model)
                        
                        sys.stdout = original_stdout
                        f.close()
            
                        model.solutions.load_from(results)
                        f = open('Testing.txt', 'w+')
                        sys.stdout = f
                        model.display()
                        sys.stdout = original_stdout
                        f.close()
            
                        file = 'Testing.txt'
                        unique2, unique1 = extractTesting(file, unique[j])
                
                        unique1 += flat_first
                
                        new_unique += [unique1, unique2]
                    elif len(last) == 1:
                        end += 1
                    for l in range(len(last)):
                        new_unique.append(last[l])
            
                else:
                    new_unique += [unique[j]]
                    end += 1
            unique = new_unique
            new_unique = []
            time += 1
            #print()
        unique = np.array(Flat(unique))
        for k in range(old_N):
            Squares[k, imp0-1, imp1-1] = np.where(unique == old_unique[k])[0][0]+1

        x.append(imp0)
        y.append(imp1)
        imp0+=1
    imp1+=1

i = 0
for i in range(old_N):
    c = Flat(Squares[i])
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=c, cmap="viridis")
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Ranking")
    ax.add_artist(legend1)
    plt.axis([-T/5, T+1, 0, T+1])
    plt.title(unique[i])
    #plt.show()
    plt.savefig("RTC/" + str(unique[i]) + '.png')
    plt.close(fig)