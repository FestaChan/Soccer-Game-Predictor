from flask import Flask, render_template, request
import pickle as p
import pandas as pd


app = Flask(__name__)


def PastPerformences(table):
    k = 4
    for i in range(table.shape[0]-1,-1,-1):
        row = table.loc[i]
        ht = row.HomeTeam
        at = row.AwayTeam

        home_ht = table.loc[table['HomeTeam'] == ht]
        away_ht = table.loc[table['AwayTeam'] == ht]
        all_h_games = home_ht.append(away_ht)
        all_h_games = all_h_games.sort_index(ascending=False)

        home_at = table.loc[table['HomeTeam'] == at]
        away_at = table.loc[table['AwayTeam'] == at]
        all_a_games = home_at.append(away_at)
        all_a_games = all_a_games.sort_index(ascending=False)

        temp_h = all_h_games.loc[i:].iloc[1:].head(k)
        temp_a = all_a_games.loc[i:].iloc[1:].head(k)
        h = len(temp_h)
        a = len(temp_a)

        if h == 0:
            h = 1

        if a == 0:
            a = 1

        table.at[i,'pastHResult'] = (temp_h[temp_h['HomeTeam'] == ht].sum().Result - \
                                     temp_h[temp_h['AwayTeam'] == ht].sum().Result)/h
        table.at[i,'pastAResult'] = (temp_a[temp_a['HomeTeam'] == at].sum().Result - \
                                     temp_a[temp_a['AwayTeam'] == at].sum().Result)/a


def assigner(table, powers):
    HAP = []
    HDP = []
    AAP = []
    ADP = []
    HSP = []
    ASP = []
    HSTP = []
    ASTP = []
    
    for index, row in table.iterrows():
        HAP.append(powers[powers['Team'] == row['HomeTeam']]['HAP'].values[0])
        HDP.append(powers[powers['Team'] == row['HomeTeam']]['HDP'].values[0])
        AAP.append(powers[powers['Team'] == row['AwayTeam']]['AAP'].values[0])
        ADP.append(powers[powers['Team'] == row['AwayTeam']]['ADP'].values[0])
        HSP.append(powers[powers['Team'] == row['HomeTeam']]['HSP'].values[0])
        ASP.append(powers[powers['Team'] == row['AwayTeam']]['ASP'].values[0])
        HSTP.append(powers[powers['Team'] == row['HomeTeam']]['HSTP'].values[0])
        ASTP.append(powers[powers['Team'] == row['AwayTeam']]['ASTP'].values[0])

    table['HAP'] = HAP
    table['HDP'] = HDP
    table['AAP'] = AAP
    table['ADP'] = ADP
    table['HSP'] = HSP
    table['ASP'] = ASP
    table['HSTP'] = HSTP
    table['ASTP'] = ASTP


@app.route('/', methods=['GET','POST'])
def home():
    winner = ['','','']
    pred = ['','','']
    msg = ""

    if request.method == 'POST':
        power_table = p.load(open("C:/Users/Festa/Desktop/machine learning/power_table.pkl",'rb'))
        recent_table = p.load(open("C:/Users/Festa/Desktop/machine learning/recent_table.pkl",'rb'))
        logreg = p.load(open("C:/Users/Festa/Desktop/machine learning/logreg.pkl",'rb'))
        XGB = p.load(open("C:/Users/Festa/Desktop/machine learning/xgb.pkl",'rb'))
        SVC = p.load(open("C:/Users/Festa/Desktop/machine learning/SVC.pkl",'rb'))

        teams = request.form.getlist("team")

        if len(teams) == 2:
            new_fixtures = pd.DataFrame([[teams[0],teams[1], 0]])
            new_fixtures.columns = ['HomeTeam','AwayTeam','Result']
            temp = new_fixtures.append(recent_table, ignore_index=True)
            temp = temp.sort_index(ascending=False)
            temp = temp.reset_index().drop(['index'],axis=1)
            test_table = temp

            PastPerformences(test_table)
            assigner(test_table, power_table)
            split = len(recent_table)
            X_test = test_table[['HAP','HDP','AAP','ADP','HSP','ASP','HSTP','ASTP','pastAResult','pastHResult']].loc[split:]
            pred[0] = logreg.predict(X_test)
            pred[1] = XGB.predict(X_test)
            pred[2] = SVC.predict(X_test)

            for i in range(3):
                if pred[i][0] == 1:
                    winner[i] = teams[0]
                elif pred[i][0] == -1:
                    winner[i] = teams[1]
                else:
                    winner[i] = 'Draw'

            msg = teams[0] + " vs. " + teams[1]
        else:
            msg = "Please Check 2 Teams"


    return render_template('soccer_page.html', prediction_logreg = winner[0], prediction_XGB = winner[1], \
        prediction_SVC = winner[2], msg = msg)


if __name__ == '__main__':
    app.run(debug=True)
