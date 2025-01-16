import requests
from bs4 import BeautifulSoup
from ScrapeWinsHelpers import Schools, links, CONVERT_NCAA, FINAL_CONVERT
import psycopg2
from config import PASSWORD

years = ['2015', '2016', '2017', '2018', '2019', '2021', '2022', '2023', '2024']
allTeams = []
def scrapewins(link, filter, year, diction):
    url = requests.get(link)
    soup = BeautifulSoup(url.text, 'html.parser')
    soup.prettify()
    table = soup.find('article')
    # Define a function to filter <ul> tags where the text in <li> elements starts with "No"
    def starts_with_no(text):
        if text is None:
            return False
        return text.startswith("No.")

    # Find all <ul> tags containing the text starting with "No" in the <li> elements
    li_tags = soup.find_all("li")
    # Filter <ul> tags where the text in <li> elements starts with "No"
    litags = [li_tag for li_tag in li_tags if starts_with_no(li_tag.get_text(separator="|", strip=True))]
    gameswithupsets = []
    for li_tag in litags:
        gameswithupsets.append(li_tag.get_text(separator="|", strip=True))
    if(filter):
        filtered_games = [game for game in gameswithupsets if starts_with_no(game.split("||")[0].strip()) and "||" in game]
        gameswithupsets = filtered_games
    
    games = []
    for game in gameswithupsets:
        games.append(game.split("||")[0].strip())
    team_names = []
    for game in games:
        # Splitting the row by commas
        parts = game.split(',')
        # Extracting the team name after "No." in the first part
        first_team_with_score = parts[0].split('No. ')[1].split(',')[0].strip()
        # Removing the score from the team name
        first_team = ' '.join(first_team_with_score.split()[:-1])
        team_names.append(first_team)
    names = [team.split(' ', 1)[1] for team in team_names]
    team_frequency = {}

    # Loop through the list of team names and count occurrences
    for team in names:
        if team in team_frequency:
            team_frequency[team] += 1
        else:
            team_frequency[team] = 1
    for team, frequency in team_frequency.items():
        inserts = [year, team, frequency]
        allTeams.append(inserts)

    for school in diction:
        if school not in team_frequency:
            inserts = [year,school, 0]
            allTeams.append(inserts)

for i in range(len(links)):
    link = links[i]
    year = years[i]
    print(link)
    print(year)
    tempdiction = {team: 0 for team in Schools[i]}
    filterCheck = False
    if year == '2021' or year == '2023' or year == '2024':
        filterCheck = True
    scrapewins(link, filterCheck, year, tempdiction)

finalInserts = [] 
for i in range(len(allTeams)):
    name = allTeams[i][1]
    if name in CONVERT_NCAA:
        name2 = CONVERT_NCAA[name]
        name3 = name2 if name2.startswith("N.C.") else name2.replace("State", "St.")
        if name3 in FINAL_CONVERT:
            name4 = FINAL_CONVERT[name3]
            teamyear = str(name4) + "_" + str(allTeams[i][0])
            finalInserts.append((teamyear, allTeams[i][2]))
        else:
            teamyear = str(name3) + "_" + str(allTeams[i][0])
            finalInserts.append((teamyear, allTeams[i][2]))
    else:
        name2 = name if name.startswith("N.C.") else name.replace("State", "St.")
        if name2 in FINAL_CONVERT:
            name3 = FINAL_CONVERT[name2]
            teamyear = str(name3) + "_" + str(allTeams[i][0])
            finalInserts.append((teamyear, allTeams[i][2]))
        else:
            teamyear = str(name2) + "_" + str(allTeams[i][0])
            finalInserts.append((teamyear, allTeams[i][2]))

connection = psycopg2.connect(
    host="localhost", port="5432", database="MarchMadness", user = "postgres", password = PASSWORD
)
crsr = connection.cursor()

# Initialize our table
crsr.execute("""CREATE TABLE IF NOT EXISTS wins(
    teamyear VARCHAR(255) PRIMARY KEY NOT NULL,
    wins INTEGER NOT NULL
);""")

connection.commit()

crsr.executemany("""
    INSERT INTO wins (
        teamyear, wins
    ) VALUES (%s, %s)
    ON CONFLICT(teamyear) DO NOTHING
    """, finalInserts)

# Commit the transaction to save changes
connection.commit()

# Close the cursor and connection
crsr.close()
connection.close()

