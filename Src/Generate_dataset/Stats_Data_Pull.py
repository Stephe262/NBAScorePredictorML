import pandas as pd
import time
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelogs, leaguegamelog
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()

# Collect all game results for each team over the past 2 seasons
def game_results(seasons):
    season_type = 'Regular Season'
    game_results = []

    for season in seasons:
        game_log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star=season_type)
        game_log_df = game_log.get_data_frames()[0]
        game_results.append(game_log_df)

    game_results_df = pd.concat(game_results, ignore_index=True)
    game_results_df1 = pd.DataFrame(game_results_df)

    # Save game results to a CSV file
    game_results_df.to_csv(f'game_results.csv', index=False)

    return game_results_df

# Collect every active players stats for every game they played over the past 2 years
def get_player_game_logs(seasons):
    season_type = 'Regular Season'
    all_players = players.get_active_players()
    player_game_logs = []

    for player in all_players:
        for season in seasons:
            player_id = player['id']
            player_logs = playergamelogs.PlayerGameLogs(player_id_nullable=player_id, season_nullable=season, season_type_nullable=season_type)
            player_logs_df = player_logs.get_data_frames()[0]
            # Select the desired columns
            columns_to_keep = ['SEASON_YEAR', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
                               'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                               'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK',
                               'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']
            player_logs_df = player_logs_df[columns_to_keep]
            player_game_logs.append(player_logs_df)

    player_combined_df = pd.concat(player_game_logs, ignore_index=True)  # Concatenate all player logs into a combined dataframe

    # Save the combined dataframe to a CSV file
    player_combined_df.to_csv(f'player_combined_stats.csv', index=False)

    return player_combined_df

def get_coordinates(city, state):
    try:
        location = geolocator.geocode(f"{city}, {state}, USA", timeout=10)
        return location.latitude, location.longitude if location else (None, None)
    except GeocoderTimedOut:
        time.sleep(1)
        return get_coordinates(city, state)

geolocator = Nominatim(user_agent="nba_team_locator", timeout=10)

# Get all team info (city, state, etc.) and add coordinates
nba_teams_df = pd.DataFrame(teams.get_teams())
team_locations = nba_teams_df.apply(lambda row: get_coordinates(row['city'], row['state']), axis=1)
nba_teams_df['latitude'], nba_teams_df['longitude'] = zip(*team_locations)

nba_teams_df.to_csv('nba_teams.csv', index=False)

seasons = ['2023-24']
# get_player_game_logs(seasons)
game_results(seasons)
