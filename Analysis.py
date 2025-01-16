import psycopg2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from config import PASSWORD

overallScore = 0
overallNumTeams=[0,0,0,0,0]
winnercorrect = 0


def scoreBracket(df):
    df = df.applymap(lambda x: x[:-5] if isinstance(x, str) else x)
    # teamnames = [[[[[["Connecticut", "Stetson"],["Florida Atlantic", "Northwestern"]], [["San Diego St.", "UAB"],["Auburn", "Yale"]]],
    #      [[["BYU", "Duquesne"],["Illinois", "Morehead St."]], [["Washington St.", "Drake"],["Iowa St.", "South Dakota St."]]]],
    #     [[[["North Carolina", "Wagner"],["Mississippi St.", "Michigan St."]], [["Saint Mary's","Grand Canyon"],["Alabama", "Charleston"]]],
    #      [[["Clemson", "New Mexico"],["Baylor", "Colgate"]], [["Dayton", "Nevada"],["Arizona", "Long Beach St."]]]]],
    #     [[[[["Purdue", "Grambling St."],["Utah St.", "TCU"]], [["Gonzaga", "McNeese St."],["Kansas", "Samford"]]],
    #      [[["South Carolina", "Oregon"],["Creighton", "Akron"]], [["Texas", "Colorado St."],["Tennessee", "Saint Peter's"]]]],
    #     [[[["Houston", "Longwood"],["Nebraska", "Texas A&M"]], [["Wisconsin", "James Madison"],["Duke", "Vermont"]]],
    #      [[["Texas Tech", "N.C. State"],["Kentucky", "Oakland"]], [["Florida", "Colorado"],["Marquette", "Western Kentucky"]]]]]]
    # # We have a bunch of embedded lists
    # winner64names = ["Connecticut", "Northwestern", "San Diego St.", "Yale", "Duquesne", "Illinois", "Washington St.", "Iowa St.",
    #             "North Carolina", "Michigan St.", "Grand Canyon", "Alabama", "Clemson", "Baylor", "Dayton", "Arizona",
    #             "Purdue", "Utah St.", "Gonzaga", "Kansas", "Oregon", "Creighton", "Texas", "Tennessee",
    #             "Houston", "Texas A&M", "James Madison", "Duke", "N.C. State", "Oakland", "Colorado", "Marquette"]
    # winner32names = ["Connecticut", "San Diego St.", "Illinois", "Iowa St.", "North Carolina", "Alabama", "Clemson", "Arizona",
    #             "Purdue", "Gonzaga", "Creighton", "Tennessee", "Houston", "Duke", "N.C. State", "Marquette"]
    # winner16names = ["Connecticut", "Illinois", "Alabama", "Clemson", "Purdue", "Tennessee", "Duke", "N.C. State"]
    # winner8names = ["Connecticut", "Alabama", "Purdue" , "N.C. State"]
    # winner4names = ["Connecticut", "Purdue"]
    # winnername = "Connecticut"
    
    # 2023 NCAA Men's Basketball Tournament matchups
    teamnames = [[[[[["Alabama", "Texas A&M Corpus Chris"], ["Maryland", "West Virginia"]],
               [["San Diego St.", "Charleston"], ["Virginia", "Furman"]]],
              [[["Creighton", "N.C. State"], ["Baylor", "UC Santa Barbara"]],
               [["Missouri", "Utah St."], ["Arizona", "Princeton"]]]],
             [[[["Purdue", "Fairleigh Dickinson"], ["Memphis", "Florida Atlantic"]],
               [["Duke", "Oral Roberts"], ["Tennessee", "Louisiana"]]],
              [[["Kentucky", "Providence"], ["Kansas St.", "Montana St."]],
               [["Michigan St.", "USC"], ["Marquette", "Vermont"]]]]],
        [[[[["Houston", "Northern Kentucky"], ["Iowa", "Auburn"]], 
               [["Miami FL", "Drake"], ["Indiana", "Kent St."]]],
              [[["Iowa St.", "Pittsburgh"], ["Xavier", "Kennesaw St."]],
               [["Texas A&M", "Penn St."], ["Texas", "Colgate"]]]],
             [[[["Kansas", "Howard"], ["Arkansas", "Illinois"]],
               [["Saint Mary's", "VCU"], ["Connecticut", "Iona"]]],
              [[["TCU", "Arizona St."], ["Gonzaga", "Grand Canyon"]],
               [["Northwestern", "Boise St."], ["UCLA", "UNC Asheville"]]]]]]

    # Winners of each round
    winner64names = [
        "Alabama", "Maryland", "San Diego St.", "Furman",
        "Creighton", "Baylor", "Missouri", "Princeton",
        "Fairleigh Dickinson", "Florida Atlantic", "Duke", "Tennessee",
        "Kentucky", "Kansas St.", "Michigan St.", "Marquette",
        "Houston", "Auburn", "Miami FL", "Indiana",
        "Pittsburgh", "Xavier", "Penn St.", "Texas",
        "Kansas", "Arkansas","Saint Mary's", "Connecticut",
        "TCU",  "Gonzaga","Northwestern", "UCLA"
    ]

    winner32names = [
        "Alabama", "San Diego St.", "Creighton", "Princeton",
        "Florida Atlantic", "Tennessee", "Kansas St.", "Michigan St.",
        "Houston", "Miami FL", "Xavier", "Texas",
        "Arkansas", "Connecticut", "Gonzaga", "UCLA"
    ]

    winner16names = [
        "San Diego St.", "Creighton",
        "Florida Atlantic", "Miami FL",
        "Connecticut", "Gonzaga",
        "Kansas St.", "Princeton"
    ]

    winner8names = [
        "San Diego St.",
        "Florida Atlantic", 
        "Miami FL",
        "Connecticut"
    ]

    winner4names = [
        "San Diego St.", "Florida Atlantic"
    ]

    winnername = "Connecticut"  # Connecticut won the 2023 NCAA Men's Basketball Tournament

    global overallScore, overallNumTeams, winnercorrect

    teams = []
    winner64 = [0] * 32
    winner32 = [0] * 16
    winner16 = [0] * 8
    winner8 = [0] * 4
    winner4 = [0] * 2
    winner = df.loc[df['teamyear'] == winnername, 'predicted_wins'].values[0]  # Get the scalar value

    for i in range(2):
        teams.append([])
        winner4[i] = df.loc[df['teamyear'] == winner4names[i], 'predicted_wins'].values[0]  # Get scalar value
        for j in range(2):
            teams[i].append([])
            winner8[i*2 + j*1] = df.loc[df['teamyear'] == winner8names[i*2 + j*1], 'predicted_wins'].values[0]
            for k in range(2):
                teams[i][j].append([])
                winner16[i*4 + j*2 + k*1] = df.loc[df['teamyear'] == winner16names[i*4 + j*2 + k*1], 'predicted_wins'].values[0]
                for l in range(2):
                    teams[i][j][k].append([])
                    winner32[i*8 + j*4 + k*2 + l*1] = df.loc[df['teamyear'] == winner32names[i*8 + j*4 + k*2 + l*1], 'predicted_wins'].values[0]
                    for m in range(2):
                        winner64[i*16 + j*8 + k*4 + l*2 + m*1] = df.loc[df['teamyear'] == winner64names[i*16 + j*8 + k*4 + l*2 + m*1], 'predicted_wins'].values[0]
                        teams[i][j][k][l].append([])
                        for n in range(2):
                            # Get scalar value instead of array
                            val = df.loc[df['teamyear'] == teamnames[i][j][k][l][m][n], 'predicted_wins'].values[0]
                            teams[i][j][k][l][m].append(val)

    # Convert teams to numpy array after all values are scalar
    teams = np.array(teams)
    
    score = 0
    if winner == np.max(teams):
        score += 320
    numteams = [0,0,0,0,0]
    for i in range(2):
        if np.max(teams[i]) == winner4[i]:
            score+=160
            numteams[4]+=1
        for j in range(2):
            if np.max(teams[i][j]) == winner8[i*2 + j]:
                score += 80
                numteams[3]+=1
            for k in range(2):
                if np.max(teams[i][j][k]) == winner16[i*4 + j*2 + k]:
                    score += 40
                    numteams[2]+=1
                for l in range(2):
                    if np.max(teams[i][j][k][l]) == winner32[i*8 + j*4 + k*2 + l]:
                        score += 20
                        numteams[1]+=1
                    for m in range(2):
                        if np.max(teams[i][j][k][l][m]) == winner64[i*16 + j*8 + k*4 + l*2 + m*1]:
                            numteams[0]+=1
                            score += 10
    print(score)
    overallScore += score
    for i in range(5):
        overallNumTeams[i] += numteams[i]
    if winner == np.max(teams):
        winnercorrect+=1

    


class SqrtRidgeRegression(nn.Module):
    def __init__(self, input_dim, alpha):
        super(SqrtRidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.alpha = alpha

    def forward(self, x):
        # Linear combination of features first
        linear_output = self.linear(x)
        # Ensure output is positive before sqrt transformation
        return torch.sqrt(torch.relu(linear_output) + 1e-6)

    def l2_penalty(self):
        return self.alpha * torch.sum(self.linear.weight ** 2)

def cross_validate(X, y, alphas, n_splits=5, epochs=1000):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=51)
    results = {}
    
    for alpha in alphas:
        fold_losses = []
        
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
            
            input_dim = X_train.shape[1]
            model = SqrtRidgeRegression(input_dim, alpha)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train_tensor)
                loss = criterion(y_pred, torch.sqrt(y_train_tensor)) + model.l2_penalty()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred_val = model(X_val_tensor)
                val_loss = criterion(y_pred_val, torch.sqrt(y_val_tensor))
                fold_losses.append(val_loss.item())
        
        mean_loss = np.mean(fold_losses)
        results[alpha] = mean_loss
        print(f"Alpha: {alpha}, Mean Validation Loss: {mean_loss:.4f}")
    
    best_alpha = min(results, key=results.get)
    print(f"Best Alpha: {best_alpha}")
    return best_alpha

def main():
    # Database connection
    connection = psycopg2.connect(
        host="localhost", port="5432", database="MarchMadness",
        user="postgres", password=PASSWORD
    )
    
    query = "SELECT * FROM wins INNER JOIN teamrankings USING (teamyear);"
    df = pd.read_sql(query, connection)
    connection.close()
    
    other_stats = pd.read_csv('team_stats.csv')[["team_id", "KenPom Off SOS", "KenPom Def SOS"]]
    merged_df = pd.merge(df, other_stats, left_on='teamyear', right_on='team_id', how='inner')
    final_df = merged_df.drop(columns=['team_id'])
    
    train_df = final_df[~(final_df['teamyear'].astype(str).str.endswith('_2024') | final_df['teamyear'].astype(str).str.endswith('_2023'))]
    test_df = final_df[final_df['teamyear'].astype(str).str.endswith('_2022')]
    
    test_df_copy = test_df.copy()  # Keep a copy for later reference
    
    X_train = train_df.drop(columns=['teamyear', 'wins']).select_dtypes(include=[np.number]).values
    y_train = train_df['wins'].values
    X_test = test_df.drop(columns=['teamyear', 'wins']).select_dtypes(include=[np.number]).values
    y_test = test_df['wins'].values
    
    mask_train = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
    mask_test = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
    
    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_clean, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_clean, dtype=torch.float32).view(-1, 1)
    spearmanTotal = 0

    NUMSIMS = 50
    for i in range(NUMSIMS):
        # Find best alpha through cross-validation
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha = cross_validate(X_train_scaled, y_train_clean, alphas)
        
        # Train final model with best alpha
        input_dim = X_train_tensor.shape[1]
        final_model = SqrtRidgeRegression(input_dim, best_alpha)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(final_model.parameters(), lr=0.001)
        
        for epoch in range(1000):
            final_model.train()
            optimizer.zero_grad()
            y_pred_train = final_model(X_train_tensor)
            loss = criterion(y_pred_train, torch.sqrt(y_train_tensor)) + final_model.l2_penalty()
            loss.backward()
            optimizer.step()
        
        final_model.eval()

        with torch.no_grad():
            y_pred_test = final_model(X_test_tensor)
            test_loss = criterion(y_pred_test, torch.sqrt(y_test_tensor))
            
            # Convert predictions back to original scale
            y_pred_test_original = y_pred_test ** 2
            
            # Calculate metrics
            y_test_np = y_test_tensor.numpy()
            y_pred_test_np = y_pred_test_original.numpy()


            correlation, _ = spearmanr(y_pred_test_np.flatten(), y_test_np.flatten())
            
            spearmanTotal += correlation
            
            # Create predictions DataFrame
            results_df = pd.DataFrame({
                'teamyear': test_df_copy['teamyear'].values,
                'actual_wins': y_test_np.flatten(),
                'predicted_wins': y_pred_test_np.flatten()
            })
            results_df = results_df.sort_values('predicted_wins', ascending=False)
            # print("\nTop Teams by Predicted Wins:")
            # pd.set_option('display.max_rows', 500)
            # print(results_df.head(70))
            scoreBracket(results_df)
            # Calculate the square root of actual wins
            sqrt_actual = np.sqrt(y_test_np)
            # Ensure both arrays are flattened to 1D
            sqrt_actual = np.ravel(sqrt_actual)  # Flatten to 1D
            y_pred_test = y_pred_test.numpy().flatten()  # Convert tensor to numpy and flatten to 1D

            # Plot square root of actual vs predicted
            if i == -1:
                # Create two subplots side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Left plot: What the model is actually doing (actual wins vs squared predictions)
                squared_predictions = y_pred_test**2
                sns.scatterplot(ax=ax1, x=squared_predictions, y=y_test_np.flatten(), alpha=0.6)
                sns.regplot(ax=ax1, x=squared_predictions, y=y_test_np.flatten(), 
                            scatter=False, color="red")
                ax1.set_xlabel("Squared Predictions")
                ax1.set_ylabel("Actual Wins")
                ax1.set_title("Actual Relationship: Actual Wins vs Squared Predictions")
                ax1.grid(True)
                
                # Right plot: Direct comparison with quadratic fit
                sns.scatterplot(ax=ax2, x=y_pred_test, y=y_test_np.flatten(), alpha=0.6)
                
                # Generate points for quadratic fit
                x_range = np.linspace(min(y_pred_test), max(y_pred_test), 100)
                z = np.polyfit(y_pred_test, y_test_np.flatten(), 2)
                p = np.poly1d(z)
                ax2.plot(x_range, p(x_range), color='red')
                
                ax2.set_xlabel("Raw Predictions (Before Squaring)")
                ax2.set_ylabel("Actual Wins")
                ax2.set_title("Direct Comparison: Actual Wins vs Raw Predictions")
                ax2.grid(True)
                
                plt.tight_layout()
                plt.show()
    averageScore = overallScore / NUMSIMS
    for i in range(5):
        overallNumTeams[i] = overallNumTeams[i]/NUMSIMS
    averageSpear = round(spearmanTotal / NUMSIMS,3)
    print(f"Across {NUMSIMS} simulations the value for Spearman's correlation was {averageSpear}")
    print(f"The bracket scored an average of {averageScore} points")
    print(f"On average it was able to predict {overallNumTeams[0]} first round matchups correctly, {overallNumTeams[1]} teams in the Sweet 16, {overallNumTeams[2]} teams in the Elite 8, and {overallNumTeams[3]} teams in the Final 4, and {overallNumTeams[4]} teams in the NCG")
    print(f"It predicted the winner correctly {winnercorrect} times")



if __name__ == "__main__":
    main()