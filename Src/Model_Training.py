import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Function to train and save the model
def train_save_model(df, target, model_name):

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    with open(f'{model_name}.pkl', 'wb') as file:
        pickle.dump(model, file)


def main():
    ## Upload Cleaned Data
    updated_teams_data_cleaned = pd.read_csv('updated_teams_data_cleaned.csv')

    # Assume 'PTS_player' and 'PTS' are the target variables for player and team models respectively
    # train_save_model(players_df_cleaned, 'PTS_player', 'player_model')
    train_save_model(updated_teams_data_cleaned, 'PTS', 'team_model')

if __name__ == "__main__":
    main()
