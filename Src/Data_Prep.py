import pandas as pd
import numpy as np
from datetime import datetime

def distance_traveled(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    R = 3958.8  # Earth's radius in miles

    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def add_rolling_moving_averages(updated_teams_data, group_col, columns, windows):

    updated_teams_data = updated_teams_data.sort_values(by = ['zGAME_DATE', group_col])

    # Save most recent data
    latest_data = updated_teams_data.loc[updated_teams_data.groupby(group_col).tail(1).index, columns]

    # Shift appropriate columns down one
    updated_teams_data[columns] = updated_teams_data.groupby(group_col)[columns].shift(1)

    # Calculate moving averages for each column and each window size
    for column in columns:
        for window in windows:
            updated_teams_data[f'{column}_MA_{window}'] = updated_teams_data.groupby(group_col)[column] \
            .transform(lambda x: x.rolling(window, min_periods=1).mean())

        # Calculate running averages for each column
        updated_teams_data[f'{column}_Rolling_AVG'] = updated_teams_data.groupby(group_col)[column] \
        .transform(lambda x: x.expanding().mean())

    # Shift appropriate columns back up
    updated_teams_data[columns] = updated_teams_data.groupby(group_col)[columns].shift(-1)
    # Input overflow data back into DF
    updated_teams_data.loc[latest_data.index, columns] = latest_data

    # Drop NA values created by the shift
    updated_teams_data.dropna(axis=0, inplace=True)

    return updated_teams_data

def clean_prep_teams(updated_teams_data, new_players_data, nba_teams_df):
    updated_teams_data.drop(columns=['VIDEO_AVAILABLE'], inplace=True)

    ###
    # Fill in todays missing data with averages for each team
    cols_to_fill = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
                    'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']

    # Calculate the average for each team in the specified columns
    team_avg = updated_teams_data.groupby('TEAM_ID')[cols_to_fill].transform('mean')

    # Fill missing values with the team average
    updated_teams_data[cols_to_fill] = updated_teams_data[cols_to_fill].fillna(team_avg)

    # Map 'W' to 1 and 'L' to 0
    updated_teams_data['WL'] = updated_teams_data['WL'].map({'W': 1, 'L': 0})

    # Calculate the average
    average_wl = updated_teams_data.groupby('TEAM_ID')['WL'].transform('mean')

    # Fill missing values based on the average
    updated_teams_data['WL'].fillna(average_wl, inplace=True)

    # Replace values above 0.5 with 'W' and values below 0.5 with 'L'
    updated_teams_data['WL'] = updated_teams_data['WL'].apply(lambda x: 'W' if x >= 0.5 else 'L')
    ###

    updated_teams_data['zOPPONENT_SCORE'] = updated_teams_data['PTS'] - updated_teams_data['PLUS_MINUS']

    updated_teams_data['zGAME_DATE'] = pd.to_datetime(updated_teams_data['GAME_DATE'])


    # Add more cols to team_game_results
    updated_teams_data.insert(30, 'zHOME_OR_AWAY',
                                updated_teams_data['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0))
    updated_teams_data.insert(30, 'zWin/Loss', updated_teams_data['WL'].apply(lambda x: 0 if 'L' in x else 1))
    updated_teams_data.insert(30, 'zDAY_OF_WEEK', updated_teams_data['zGAME_DATE'].dt.dayofweek.astype('int64') + 1)
    updated_teams_data.insert(30, 'zFG3A%', updated_teams_data['FG3A'] / updated_teams_data['FGA'])
    updated_teams_data.insert(30, 'zPTS/FGA', updated_teams_data['PTS'] / updated_teams_data['FGA'])
    updated_teams_data.insert(30, 'zFTA/FGA', updated_teams_data['FTA'] / updated_teams_data['FGA'])
    updated_teams_data.insert(30, 'zGAME_LOCATION', updated_teams_data['MATCHUP'].apply(
        lambda x: x.split('@')[1].strip() if '@' in x else x.split('vs.')[0].strip()))

    # Add Latitude and Longitute for game_location (join on nba_teams df)
    updated_teams_data = pd.merge(updated_teams_data, nba_teams_df[['abbreviation', 'latitude', 'longitude']],
                                    left_on='zGAME_LOCATION', right_on='abbreviation')
    updated_teams_data.rename(columns={'abbreviation': 'GAME_LOCATION'}, inplace=True)

    # Sort the merged_df based on PLAYER_ID and GAME_ID
    updated_teams_data.sort_values(by=['TEAM_ID', 'GAME_DATE'], inplace=True)

    # Add days_between_games and distance_traveld using sorted DF
    updated_teams_data.insert(33, 'zDAYS_BETWEEN_GAMES',
                                updated_teams_data.groupby('TEAM_ID')['GAME_DATE'].diff().apply(lambda x: x.days))
    updated_teams_data['zDISTANCE_TRAVELED'] = updated_teams_data.groupby('TEAM_ID').apply(
        lambda x: distance_traveled(x['latitude'].shift(), x['longitude'].shift(), x['latitude'],
                                    x['longitude'])).reset_index(level=0, drop=True).fillna(0)


    # Replace 'inf' with NaN and then drop rows with NaN
    updated_teams_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    updated_teams_data.dropna(inplace=True)

    updated_teams_data = updated_teams_data[['zGAME_DATE', 'GAME_ID', 'TEAM_ID', 'TEAM_NAME',
                                                 'MATCHUP', 'FGM', 'FGA', 'FG_PCT', 'FG3M',
                                                 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB',
                                                 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
                                                 'PTS', 'PLUS_MINUS', 'zOPPONENT_SCORE', 'zDAYS_BETWEEN_GAMES',
                                                 'zFTA/FGA', 'zPTS/FGA', 'zFG3A%', 'zDAY_OF_WEEK', 'zWin/Loss', 'zHOME_OR_AWAY',
                                                 'GAME_LOCATION', 'zDISTANCE_TRAVELED']].copy()
    # Columns for which to calculate moving averages
    columns = ['FGM', 'FGA', 'FG_PCT', 'FG3M',
               'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB',
               'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
               'PTS', 'PLUS_MINUS', 'zOPPONENT_SCORE', 'zDAYS_BETWEEN_GAMES',
               'zFTA/FGA', 'zPTS/FGA', 'zFG3A%', 'zWin/Loss', 'zDISTANCE_TRAVELED']

    # window sizes for rolling averages
    windows = [1, 3, 7, 14, 30]

    updated_teams_data = add_rolling_moving_averages(updated_teams_data, 'TEAM_ID', columns, windows)

    updated_teams_data.sort_values('zGAME_DATE', inplace=True)

    # One-hot encode the game locations
    ohe = OneHotEncoder(drop='first')
    ohe_df = pd.DataFrame(ohe.fit_transform(updated_teams_data[['GAME_LOCATION']]).toarray())
    ohe_df.columns = ohe.get_feature_names_out()

    # Reset the indices of both dataframes
    updated_teams_data.reset_index(drop=True, inplace=True)
    ohe_df.reset_index(drop=True, inplace=True)

    # Concatenate the encoded columns to the original dataframe
    updated_teams_data = pd.concat([updated_teams_data, ohe_df], axis=1)

    # Drop columns so we only use lagging data
    updated_teams_data.drop(columns=['FGM', 'FGA', 'FG_PCT', 'FG3M',
                                       'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB',
                                       'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
                                       'PLUS_MINUS', 'zFTA/FGA', 'zPTS/FGA',
                                       'zFG3A%'], inplace=True)

    # Before dropping columns, save the required columns into a separate DataFrame
    saved_team_columns = updated_teams_data[['GAME_ID', 'zGAME_DATE', 'TEAM_NAME', 'zWin/Loss', 'MATCHUP', 'zOPPONENT_SCORE']].copy()
    saved_team_columns['Opponent'] = updated_teams_data['MATCHUP'].str[-3:]

    # Remove team info
    updated_teams_data = updated_teams_data.drop(['GAME_ID', 'TEAM_ID', 'TEAM_NAME',
                                                      'MATCHUP', 'zGAME_DATE', 'zWin/Loss', 'zOPPONENT_SCORE',
                                                      'GAME_LOCATION'], axis=1).copy()


    updated_teams_data.to_csv('updated_teams_data.csv')
    saved_team_columns.to_csv('saved_team_columns.csv')
    return (updated_teams_data, saved_team_columns)

def clean_prep_players(players_df, team_game_results_df, nba_teams_df):

    # Combine players_df and team_game_results
    players_df = pd.merge(players_df, team_game_results_df, on=['GAME_ID', 'TEAM_ID'], suffixes=('_player', '_team'))

    # Drop unneccesary columns
    players_df.drop(columns=['TEAM_ABBREVIATION_team', 'TEAM_NAME_team', 'GAME_DATE_team', 'MATCHUP_team', 'WL_team',
                             'VIDEO_AVAILABLE'], inplace=True)
    # Rename columns
    players_df.rename(columns={'TEAM_ABBREVIATION_player': 'TEAM_ABBREVIATION', 'TEAM_NAME_player': 'TEAM_NAME',
                               'GAME_DATE_player': 'GAME_DATE', 'MATCHUP_player': 'MATCHUP', 'WL_player': 'WL',
                               'POSS': 'POSS_team'}, inplace=True)
    # Add 'OPPONENT_SCORE' column
    players_df['zOPPONENT_SCORE'] = players_df['PTS_team'] - players_df['PLUS_MINUS_team']

    # change 'GAME_DATE' to datetime format
    players_df['zGAME_DATE'] = pd.to_datetime(players_df['GAME_DATE'])
    print(players_df['WL'])


    # Add more cols to merged_df
    players_df.insert(9, 'zHOME_OR_AWAY', players_df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0))
    players_df.insert(11, 'zWin/Loss', players_df['WL'].apply(lambda x: 0 if 'L' in x else 1))
    players_df.insert(1, 'Season_Start_Year', players_df['SEASON_YEAR'].str[:4].astype(int))
    players_df.insert(8, 'zDAY_OF_WEEK', players_df['zGAME_DATE'].dt.dayofweek.astype('int64') + 1)
    players_df.insert(26, 'zFG3A%', players_df['FG3A_player'] / players_df['FGA_player'])
    players_df.insert(27, 'zPTS/FGA', players_df['PTS_player'] / players_df['FGA_player'])
    players_df.insert(28, 'zFTA/FGA', players_df['FTA_player'] / players_df['FGA_player'])
    players_df.insert(29, 'zEFG%',
                      ((players_df['FGM_player'] + (1.5 * players_df['FG3M_player'])) / players_df['FGA_player']))
    players_df.insert(30, 'zTS%',
                      players_df['PTS_player'] / (2 * (players_df['FGA_player'] + 0.44 * players_df['FTA_player'])))
    players_df.insert(31, 'zAST_TOV', players_df['AST_player'] / players_df['TOV_player'])
    players_df.insert(11, 'zGAME_LOCATION', players_df['MATCHUP'].apply(
        lambda x: x.split('@')[1].strip() if '@' in x else x.split('vs.')[0].strip()))


    # Add Latitude and Longitute for game_location (join on nba_teams df)
    players_df = pd.merge(players_df, nba_teams_df[['abbreviation', 'latitude', 'longitude']], left_on='zGAME_LOCATION',
                          right_on='abbreviation')

    # Sort the merged_df based on PLAYER_ID and GAME_DATE
    players_df.sort_values(by=['PLAYER_ID', 'GAME_DATE'], inplace=True)

    # Add days_between_games and distance_traveld using sorted DF
    players_df.insert(10, 'zDAYS_BETWEEN_GAMES',
                      players_df.groupby('PLAYER_ID')['zGAME_DATE'].diff().apply(lambda x: x.days))
    players_df['zDISTANCE_TRAVELED'] = players_df.groupby('PLAYER_ID').apply(
        lambda x: distance_traveled(x['latitude'].shift(), x['longitude'].shift(), x['latitude'],
                                    x['longitude'])).reset_index(level=0, drop=True).fillna(0)


    # Add days_between_games and distance_traveld using sorted DF
    players_df = players_df.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID']).reset_index().drop('index', axis=1)

    # Add new cols from merged_df back to players_df
    selected_columns = ['PLAYER_ID', 'GAME_ID', 'zWin/Loss', 'zFG3A%', 'zPTS/FGA', 'zFTA/FGA', 'zEFG%', 'zTS%', 'zAST_TOV',
                        'zDISTANCE_TRAVELED']

    players_df = players_df.merge(players_df[selected_columns], how='inner')

    # Replace 'inf' with NaN and then drop rows with NaN
    players_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    players_df.dropna(inplace=True)

    # Columns for which to calculate moving averages
    columns = ['zWin/Loss', 'MIN_player', 'FGM_player', 'FGA_player', 'FG_PCT_player',
               'FG3M_player', 'FG3A_player', 'FG3_PCT_player', 'FTM_player',
               'FTA_player', 'FT_PCT_player', 'OREB_player', 'DREB_player',
               'REB_player', 'AST_player', 'TOV_player', 'STL_player', 'BLK_player',
               'BLKA', 'PF_player', 'PFD', 'PTS_player',
               'PLUS_MINUS_player', 'zDISTANCE_TRAVELED']

    # window sizes for rolling averages
    windows = [1, 3, 7, 14, 30]

    # Now you can call the function with any DataFrame like this:
    players_df = add_rolling_moving_averages(players_df, 'PLAYER_ID', columns, windows)
    players_df.sort_values('PLAYER_ID').head()

    # Drop unneccesary columns as I will only look at lagging data
    players_df.drop(columns=['SEASON_ID', 'Season_Start_Year',
                             'PLAYER_ID', 'SEASON_YEAR', 'TEAM_ABBREVIATION', 'abbreviation', 'TEAM_ID',
                             'zWin/Loss', 'MIN_player', 'FGM_player', 'zGAME_DATE',
                             'FGA_player', 'FG_PCT_player', 'FG3M_player', 'FG3A_player',
                             'FG3_PCT_player', 'FTM_player', 'FTA_player', 'FT_PCT_player',
                             'OREB_player', 'DREB_player', 'REB_player', 'AST_player', 'TOV_player',
                             'STL_player', 'BLK_player', 'BLKA', 'PF_player', 'PFD',
                             'PLUS_MINUS_player'], axis=1, inplace=True)

    # One-hot encode the game locations
    ohe = OneHotEncoder(drop='first')
    ohe_df = pd.DataFrame(ohe.fit_transform(players_df[['zGAME_LOCATION']]).toarray())
    ohe_df.columns = ohe.get_feature_names_out()

    # Reset the indices of both dataframes
    players_df.reset_index(drop=True, inplace=True)
    ohe_df.reset_index(drop=True, inplace=True)

    # Concatenate the encoded columns to the original dataframe
    players_df = pd.concat([players_df, ohe_df], axis=1)

    # Before dropping columns, save the required columns into a separate DataFrame
    saved_players_columns = players_df[['GAME_DATE', 'PLAYER_NAME', 'TEAM_NAME', 'GAME_ID', 'WL']].copy()
    saved_players_columns['Opponent'] = players_df['MATCHUP'].str[-3:]

    # Now drop all other columns not used for model training
    players_df = players_df.drop(['zGAME_LOCATION', 'TEAM_NAME', 'PLAYER_NAME', 'GAME_DATE',
                                  'MATCHUP', 'WL'], axis=1).copy()

    return (players_df, saved_players_columns)


def main():
    ## Upload Data
    updated_teams_data = pd.read_csv('game_results.csv')
    new_players_data = pd.read_csv('player_combined_stats.csv')
    nba_teams_df = pd.read_csv('nba_teams.csv')

    # players_df_cleaned, saved_players_columns = clean_prep_players(players_df, team_game_results_df, nba_teams_df)
    updated_teams_data_cleaned, saved_teams_columns = clean_prep_teams(updated_teams_data, new_players_data, nba_teams_df)

if __name__ == "__main__":
    main()
