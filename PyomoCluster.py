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

file = "Data/Tennis/tennis_atp-master/atp_matches_"
formatDate = '%Y%m%d'
date = dt.datetime(2020, 1, 1)
data =  np.array(ReadYear(file, date.year))
d = len(data)
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

#read = read[0:130]
t = 100
i = 0
"""
unique = np.array(real_rank)
data = read[:, 1:6]
print(data)
"""
"""
data = [
    ["a", "b", "1", "0", "cup"],
    ["a", "d", "1", "0", "cup"],
    ["b", "c", "1", "0", "cup"],
    ["b", "d", "1", "0", "cup"],
    ["c", "a", "1", "0", "cup"],
    ["d", "c", "1", "0", "cup"]
]
"""
"""
d = len(data)
data, d = KeepOnlyGames(data, real_rank, d)
"""
new_unique = []
unique = np.array(Countries(data))
#print(unique)
#unique = [["a", "b", "c", "d"]]
print(d)
TestMatrix = MatchTable(data, unique, len(unique))
"""
m = np.array([[0,1,1,0],
 [0,0,1,1],
 [0,0,0,1],
 [1,0,0,0]])
"""
#matrix = m.copy()
matrix = np.array(MatchTable(data, unique, len(unique)))
m = np.array(MatchTable(data, unique, len(unique)))
UN = len(unique)
N = len(unique)
"""
r, unique, m, matrix, N, data, d = ExtractRankingFile("fifa_ranking.csv", unique, m, matrix, date, data, d)
"""
#test = unique
matrix, m, unique, N, last = RemoveNoWin(matrix, m, unique, N)
#matrix, m, unique, N, first = RemoveOnlyWin(matrix, m, unique, N)
fl = Flat(last)
#flat_first = Flat(first)
data, d, removed_games = RemoveGames(data, fl, d)
#data, d, removed_games = RemoveGames(data, flat_first, d)

test = unique
unique = [unique]
TestMatrix = m.copy()
#matrix = matrix.tolist()

csv = "Matrix.csv"

WriteCSV(csv, matrix, unique[0], N)
end = 0
count = 0

start = time.time()

while end != len(unique) and end < UN:
    end = 0
    for j in range(len(unique)):
        N = len(unique[j])
        if N > 1:
            #data = read[:, 1:6]
            
            new_data, new_d = KeepOnlyGames(data, np.array(unique[j]), d)

            #matrix = np.array(MatchTable(data, unique, N)).astype(np.float64)
            matrix = np.array(MatchTable(new_data, np.array(unique[j]), N))
            m = np.array(MatchTable(new_data, np.array(unique[j]), N))
            #print(matrix)
            #new_d = d
            #flat_last = []
            #flat_first = []
            
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
                    #return pyo.summation(m.c, m.x)
                    return Clusters(matrix, model.r, unique[j], N, new_d)

                model.OBJ = pyo.Objective(rule=obj_expression)

                def constr(model):
                    return sum(model.r[i] for i in model.I) >= 1

                
                model.constraints = pyo.ConstraintList()
                model.constraints.add(expr = sum(model.r[i] for i in model.I) >= int(N/2))
                model.constraints.add(expr = sum(model.r[i] for i in model.I) <= int((N+1)/2))
                """
                
                model.constraints = pyo.ConstraintList()
                model.constraints.add(expr = sum(model.r[i] for i in model.I) >= 1)
                model.constraints.add(expr = sum(model.r[i] for i in model.I) <= N-1)
                """
                #print(model.constraints)
                #model.constr = pyo.Constraint(rule=constr)
                original_stdout = sys.stdout
                f = open('Warning' + str(count) + '.txt', 'w+')
                sys.stdout = f
                solver = pyo.SolverFactory('cplex')
                results = solver.solve(model)
                sys.stdout = original_stdout
                f.close()
            
                model.solutions.load_from(results)
                
                f = open('Testing' + str(count) + '.txt', 'w+')
                sys.stdout = f
                model.display()
                sys.stdout = original_stdout
                f.close()
            
                file = 'Testing' + str(count) + '.txt'
                unique2, unique1 = extractTesting(file, unique[j])
                unique1 += flat_first
                new_unique += [unique1, unique2]
                count += 1
            elif len(last) == 1:
                end += 1
            for l in range(len(last)):
                new_unique.append(last[l])
            
        else:
            end += 1
            new_unique += [unique[j]] 
    #print(unique)
    unique = new_unique
    new_unique = []
    i += 1
    #print(unique)
    #print(end)
    #print(len(unique))

print("time: " + str(time.time() - start))
unique = Flat(unique)
unique += fl
unique = np.array(unique)
print(len(unique))
print(len(test))
test = np.array(test)
print(unique)
print(np.sum(TestMatrix))
print(Upsets(TestMatrix, unique, test, len(test), d))

