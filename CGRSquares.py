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
ldata = len(data)

data, ldata = KeepOnlyGames(data, real_rank, ldata)
print(ldata)

unique  = Countries(data)
N = len(unique)
print(N)

imp1 = 1
G = 50
Squares = np.zeros([N,G,G])
x = []
y = []
while imp1 <= G:
    imp0 = 1
    #print(str(((imp1-1)*G+(imp0-1))/(G*G)*100) + "%")
    while imp0 <= G:
        print(str(((imp1-1)*G+(imp0-1))/(G*G)*100) + "%")
        importance = [imp0, imp1]
        matrix, draws = np.array(MatchTableDrawsImportance(data, unique, N, importance))

        best_upsets = 100000
        step = 1
        rho = 0.9
        b1 = 0.2
        b2 = 0.21
        phi = 1
        T = 10
        
        scores = np.zeros(T)
        upsets_plt = np.zeros(T)
        g_plt = np.zeros(T)
        rank_score = np.zeros(T)

        ranking = []
        new_gradient = []
        gradient = True
        if gradient == True:
            #j = 0
            score_rank = np.zeros(N)
            # get the teams that do not have wins or draws
            #new_matrix, new_m, new_unique, new_N, last = RemoveNoWinDraws(matrix, draws, m, unique, N)
            #flat_last = Flat(last)
            last = []
            # set the last's score
            score_rank = SetScore(last, matrix, unique, score_rank, N, ldata)
            score_rank = UpdateScore(last, unique, score_rank)
            step = 10/np.linalg.norm(Score_deriv(matrix, draws, score_rank, unique, N))
            # set the score of other teams
            """
            rank_score = init_value
            rank_score = rank_score.tolist()
            
            vec = score_rank
            sorted_indices = vec.argsort()
            ranked = [(unique[i], vec[i]) for i in sorted_indices]
            ranked.reverse()
            rank = [tple[0] for tple in ranked]
            """
            #while rank_score[0] < 10 and rank_score[-len(flat_last)-1] > len(last)+1:
            for t in range(T):
                print(str(t/T))
                #new_read = read
                new_data = data
                new_unique = Countries(data)
                new_N = len(new_unique)
                new_d = len(new_data)
                
                new_matrix, new_draws = np.array(MatchTableDrawsImportance(new_data, new_unique, new_N, importance))

                # compute gradient
                grad = new_Score_deriv(new_matrix, new_draws, score_rank, new_unique, new_unique)
                #print("gradient: " + str(np.linalg.norm(grad)))
        
                # update the step
                f = sum(new_Score(matrix, new_draws, score_rank, unique, new_unique))
                #print("score: " + str(f))
                #grad = new_gradient
                d = np.sign(grad)
                new_score_rank = UpdateScore(last, unique, score_rank + step * grad)
                new_f = sum(new_Score(matrix, new_draws, new_score_rank, unique, new_unique))
                new_grad = new_Score_deriv(matrix, new_draws, new_score_rank, unique, new_unique)
                c = 0
                while conditions(f, grad, new_f, new_grad, d, step, b1, b2):
                    step, new_grad = Wolfe(matrix, draws, unique, new_unique, last, score_rank, grad, new_grad, d, step, rho, b2)
                    step, new_f = Armijo(matrix, draws, unique, new_unique, last, score_rank, f, grad, new_f, d, step, rho, b1)
                    c += 1;
                    if c > 1:
                        break

                #print("step: " + str(step))
                #print(new_gradient)
                score_rank += grad*step
                # set back the last's scores
                score_rank = UpdateScore(last, unique, score_rank)
                """
                vec = score_rank
                sorted_indices = vec.argsort()
                ranked = [(unique[i], vec[i]) for i in sorted_indices]
                ranked.reverse()
                rank = [tple[0] for tple in ranked]
                rank_score = [tple[1] for tple in ranked]
                """
                #upsets = Upsets(m, rank, unique, N, d)

                #upsets_plt[t] = upsets)
                """
                if upsets < best_upsets:
                    best_upsets = upsets
                    ranking = rank
                    print("j: " + str(j))
                    print("upsets: " + str(upsets))
                    print(ranking[:10])
                    r_score = rank_score[:10]
                    print(r_score)
                """
                #j += 1

            #upsets = Upsets(m, rank, unique, N, ldata)
        vec = score_rank
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
                        loc="lower left", title="Ranking")
    ax.add_artist(legend1)
    plt.axis([-G/5, G+1, 0, G+1])
    plt.title(unique[i])
    #plt.show()
    plt.savefig("CGR/" + str(unique[i]) + '.png')
    plt.close(fig)
