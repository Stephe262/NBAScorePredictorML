import pandas as pd
import numpy as np
import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error
from Data_Prep import clean_prep_teams
from Pull_Todays_Games import add_todays_games
from datetime import datetime
from Stats_Data_Pull import game_results

# Function to load trained models
def load_model(model_name):
    with open(f'{model_name}.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict_points_score_v3(df, saved_df, target, model_name, file_name):
    # Store true values in separate variables
    y_test_df = df[target]

    # Drop target to make predictions
    df_target = df.drop(target, axis=1)

    # Generate predictions
    df_results = model_name.predict(df_target)
    df_results = pd.to_numeric(df_results, errors='coerce')

    # Merge saved columns with predictions
    output_df = saved_df.copy()
    output_df['Actual_Points'] = y_test_df
    output_df['Predicted_Points'] = df_results


    # Check if the data is for teams or players
    if 'PLAYER_NAME' not in output_df.columns:
        # Separate home and away teams
        home_df = output_df[output_df['MATCHUP'].str.contains('vs.')].copy()
        away_df = output_df[output_df['MATCHUP'].str.contains('@')].copy()

        # Rename columns for merging
        home_df.rename(columns={'TEAM_NAME': 'HOME_TEAM', 'Actual_Points': 'HOME_TEAM_ACTUAL_POINTS',
                                'Predicted_Points': 'HOME_TEAM_PRED_POINTS', 'zGAME_DATE': 'GAME_DATE'}, inplace=True)
        away_df.rename(columns={'TEAM_NAME': 'AWAY_TEAM', 'Actual_Points': 'AWAY_TEAM_ACTUAL_POINTS',
                                'Predicted_Points': 'AWAY_TEAM_PRED_POINTS'}, inplace=True)

        # Merge home and away dataframes
        merged_df = pd.merge(home_df[['GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'HOME_TEAM_ACTUAL_POINTS', 'HOME_TEAM_PRED_POINTS']],
                    away_df[['GAME_ID', 'AWAY_TEAM', 'AWAY_TEAM_ACTUAL_POINTS', 'AWAY_TEAM_PRED_POINTS']],
                    on='GAME_ID', how='inner')

        # Add columns for actual and predicted winning team
        merged_df['ACTUAL_WINNING_TEAM'] = merged_df.apply(
            lambda x: x['HOME_TEAM'] if x['HOME_TEAM_ACTUAL_POINTS'] > x['AWAY_TEAM_ACTUAL_POINTS'] else x['AWAY_TEAM'],
            axis=1)
        merged_df['PRED_WINNING_TEAM'] = merged_df.apply(
            lambda x: x['HOME_TEAM'] if x['HOME_TEAM_PRED_POINTS'] > x['AWAY_TEAM_PRED_POINTS'] else x['AWAY_TEAM'],
            axis=1)

        merged_df['ACTUAL_LOSING_TEAM'] = merged_df.apply(
            lambda x: x['HOME_TEAM'] if x['HOME_TEAM_ACTUAL_POINTS'] < x['AWAY_TEAM_ACTUAL_POINTS'] else x['AWAY_TEAM'],
            axis=1)
        merged_df['PRED_LOSING_TEAM'] = merged_df.apply(
            lambda x: x['HOME_TEAM'] if x['HOME_TEAM_PRED_POINTS'] < x['AWAY_TEAM_PRED_POINTS'] else x['AWAY_TEAM'],
            axis=1)

        # Add a column to indicate if the prediction was correct
        merged_df['CORRECT_WIN'] = (merged_df['ACTUAL_WINNING_TEAM'] == merged_df['PRED_WINNING_TEAM']).astype(int)
        merged_df['CORRECT_LOSE'] = (merged_df['ACTUAL_LOSING_TEAM'] == merged_df['PRED_LOSING_TEAM']).astype(int)

        merged_df['HOME_TEAM_PRED_POINTS'] = pd.to_numeric(merged_df['HOME_TEAM_PRED_POINTS'], errors='coerce')
        merged_df['AWAY_TEAM_PRED_POINTS'] = pd.to_numeric(merged_df['AWAY_TEAM_PRED_POINTS'], errors='coerce')

        # Select and order the columns as required
        final_output_df = merged_df[
            ['GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'HOME_TEAM_ACTUAL_POINTS', 'HOME_TEAM_PRED_POINTS', 'AWAY_TEAM',
             'AWAY_TEAM_ACTUAL_POINTS', 'AWAY_TEAM_PRED_POINTS', 'ACTUAL_WINNING_TEAM', 'PRED_WINNING_TEAM',
             'CORRECT_WIN', 'ACTUAL_LOSING_TEAM', 'PRED_LOSING_TEAM', 'CORRECT_LOSE']]

        # Sort by game date
        final_output_df.sort_values(by=['GAME_DATE', 'GAME_ID'], inplace=True)
        # Save the final output DataFrame to a CSV file
        final_output_df.to_csv(f'{file_name}.csv', index=False)

        ### Create new XLSX file to be used in Tableau ###
        formatted_data = []

        # Iterate through each game in the merged DataFrame
        for _, row in merged_df.iterrows():
            # Add the home team row
            formatted_data.append({
                'GAME_ID': row['GAME_ID'],
                'GAME_DATE': row['GAME_DATE'],
                'Team': row['HOME_TEAM'],
                'HOME/AWAY': 'Home',
                'Points - Actual': row['HOME_TEAM_ACTUAL_POINTS'],
                'Points - Pred': row['HOME_TEAM_PRED_POINTS'],
                'W/L - Pred': 'Win' if row['HOME_TEAM_PRED_POINTS'] > row['AWAY_TEAM_PRED_POINTS'] else 'Lose',
                'W/L - Actual': 'Win' if row['HOME_TEAM_ACTUAL_POINTS'] > row['AWAY_TEAM_ACTUAL_POINTS'] else 'Lose',
            })

            # Add the away team row
            formatted_data.append({
                'GAME_ID': row['GAME_ID'],
                'GAME_DATE': row['GAME_DATE'],
                'Team': row['AWAY_TEAM'],
                'HOME/AWAY': 'Away',
                'Points - Actual': row['AWAY_TEAM_ACTUAL_POINTS'],
                'Points - Pred': row['AWAY_TEAM_PRED_POINTS'],
                'W/L - Pred': 'Win' if row['AWAY_TEAM_PRED_POINTS'] > row['HOME_TEAM_PRED_POINTS'] else 'Lose',
                'W/L - Actual': 'Win' if row['AWAY_TEAM_ACTUAL_POINTS'] > row['HOME_TEAM_ACTUAL_POINTS'] else 'Lose',
            })

        # Convert the list of formatted data to a DataFrame
        formatted_output_df = pd.DataFrame(formatted_data)

        # Reorder the DataFrame columns to match the desired output
        formatted_output_df = formatted_output_df[['GAME_ID', 'GAME_DATE', 'Team', 'HOME/AWAY',
                                                   'Points - Actual', 'Points - Pred', 'W/L - Pred', 'W/L - Actual']]

        # Save the formatted output to a XLXS file to be used in Tablea for visualizations
        formatted_output_df.to_excel('Tableau/NBA_Tableau_Data_v2.xlsx', index=False)

    else:
        output_df.to_csv(f'{file_name}.csv', index=False)

    # Calculate and print RMSE for Team Point Predictions
    rmse_df = sqrt(mean_squared_error(y_test_df, df_results))
    print(f'{file_name} RMSE: {rmse_df}')



# ------------------------------------------------------------------------------------------------------------ #
# Main function
def main():
  """ USE BELOW IF YOU ALREADY HAVE A TRAINED MODEL AND WANT TO CONTINUE PREDICTING ON THE CURRENT SEASON """
    # # Load in Data for current NBA Season
    # seasons = ['2023-24']
    # new_teams_data = game_results(seasons)

    # # new_players_data = pd.read_csv('player_combined_stats.csv')
    # nba_teams_df = pd.read_csv('nba_teams.csv')
    
    # # Add today's games to season so far
    # updated_teams_data = add_todays_games(new_teams_data)

    # # Preprocess new data
    # # preprocessed_players_data, saved_players_columns = clean_prep_players(new_players_data, updated_teams_data, nba_teams_df)
    # preprocessed_teams_data, saved_teams_columns= clean_prep_teams(updated_teams_data, new_players_data, nba_teams_df)
""" ------------------------------------------------------------------------------------------------------------ """

    # Load models
    # player_model = load_model('player_model')
    team_model = load_model('team_model')

    # Load cleaned data
    preprocessed_teams_data = pd.read_csv('updated_teams_data_cleaned.csv')
    saved_teams_columns = pd.read_csv('saved_team_columns.csv)
    
    # Predict and output to CSV
    predict_points_score_v3(preprocessed_teams_data, saved_teams_columns, 'PTS', team_model, 'team_predictions')
    # predict_points_score_v3(preprocessed_players_data, saved_players_columns, 'PTS_player', player_model, 'player_predictions')

if __name__ == "__main__":
    main()
