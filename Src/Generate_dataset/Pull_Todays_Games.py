import json
import pandas as pd
from datetime import datetime
import requests


def add_todays_games(new_teams_data):
    new_teams_data['GAME_DATE'] = pd.to_datetime(new_teams_data['GAME_DATE'], format='%Y-%m-%d')
    # Get today's date in the required format
    todays_date = datetime.today().strftime('%m/%d/%Y 00:00:00')

    # URL to fetch NBA schedule data
    url = 'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json'

    # Send a GET request to the URL
    response = requests.get(url)

    # Initialize lists to store data
    data_rows = []

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data
        data = response.json()
        # Access today's matchups
        game_dates = data.get("leagueSchedule").get("gameDates")

        for game_date in game_dates:
            if game_date.get("gameDate") == todays_date:
                matchups = game_date.get("games")
                if matchups:
                    for matchup in matchups:
                        home_team = matchup.get("homeTeam")
                        away_team = matchup.get("awayTeam")
                        game_id = matchup.get("gameId")

                        # Create a row for the home team
                        home_row = [
                            22023, ## Note that this is the current season's Season_Id, fill in with appropriate season
                            home_team.get("teamId"),
                            home_team.get("teamTricode"),
                            f'{home_team.get("teamCity")} {home_team.get("teamName")}',
                            game_id[2:],
                            datetime.today().strftime('%m/%d/%Y'),
                            f"{home_team.get('teamTricode')} vs. {away_team.get('teamTricode')}"
                        ]

                        # Create a row for the away team
                        away_row = [
                            22023, ## Note that this is the current season's Season_Id, fill in with appropriate season
                            away_team.get("teamId"),
                            away_team.get("teamTricode"),
                            f'{away_team.get("teamCity")} {away_team.get("teamName")}',
                            game_id[2:],
                            datetime.today().strftime('%m/%d/%Y'),
                            f"{away_team.get('teamTricode')} @ {home_team.get('teamTricode')}"
                        ]

                        # Append the rows to the data list
                        data_rows.append(home_row)
                        data_rows.append(away_row)

        # Create the dataframe
        todays_games_df = pd.DataFrame(data_rows, columns=['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP'])
        todays_games_df['GAME_DATE'] = pd.to_datetime(todays_games_df['GAME_DATE'], format='%m/%d/%Y')
    else:
        print("Failed to fetch NBA schedule data.")
    joined_df = pd.concat([new_teams_data, todays_games_df], ignore_index=True, sort=False)
    return joined_df

