import requests
from bs4 import BeautifulSoup
from QualityControl import convertTR, percentToFloat, perGameToPlay, PercentFTToPlay, numPlays, maptoIndex
import psycopg2
import time
from config import PASSWORD

# Beginning at the homepage, extract links for all of our statistics
url = requests.get('https://www.teamrankings.com/ncb/team-stats/')
soup = BeautifulSoup(url.text, 'html.parser')
soup.prettify()
table = soup.find('div', attrs = {'class' : 'main-wrapper clearfix has-left-sidebar'}).find('main')
links = table.find_all('a')

# Selecting the years that we want - because of kenpom restrictions we must include the stats from tournament games as well
# Route 1: Do not include tournament, and have mid major teams with propped up Strength of Schedule in relation to their stats
# Route 2: Include tournament, and have those tournament stats effect the averages, and thus become slightly correlated with tourney wins
# Route 2 is what I chose, because I would not expect the factors for a team to win in March to be OVERLY different from a regular season game
#       Just provides more data on the type of team they are
dates = ['2010-05-01', '2011-05-01', '2012-05-01', '2013-05-01', '2014-05-01', '2015-05-01', '2016-05-01', '2017-05-01',
         '2018-05-01', '2019-05-01', '2021-05-01', '2022-05-01', '2023-05-01', '2024-05-01']
category = "empty"


# Selecting relevant statistics (looking to avoid multicollinearity)

# We will take the per Game stats, and convert them into per Offensive Play/Defensive Play
stats = ["Points per Game", "Points from 2 pointers", "Points from 3 pointers", "Percent of Points from Free Throws",
         "Two Point %", "Three Point %", "Free Throw %", "Offensive Rebounding %", "Defensive Rebounding %", 
         "Total Rebounding % (Rebound Rate)", "Steals per Game", "Steals per Defensive Play", "Blocks per Game", "Assists per Game", 
         "Turnovers per Offensive Play", "Turnovers per Game", "Personal Fouls per Game", "Games Played", "Possessions per Game"]

# We will store the stats for each year of a team in this dictionary teams (i.e. 2007 Duke)
teams = {}

teamstoskip = ["Cornell_2021", "Maryland Eastern Shore_2021", "Penn_2021", "Dartmouth_2021", "Bethune Cookman_2021", "Princeton_2021",
                 "Columbia_2021", "Brown_2021", "Harvard_2021", "Yale_2021"]
counter = 0
for date in dates:
    # Extract year from our date for statkeeping purposes
    year = date[:4]
    counter+=1
    if counter % 9 == 0:
        print("5 minute timer started")
        time.sleep(300)
    for link in links:

        # Extract the link, initialize the header (if necessary), skip if not in the included stat list
        linkstring = link.get('href')
        if linkstring == '#':
            category = link.get_text(strip=True)
            continue
        else:
            tablename = link.get_text(strip=True)
            if tablename[0:8] == "1st Half" or tablename[0:8] == "Overtime" or tablename[0:8] == "2nd Half":
                continue
            noOpp = tablename[9:]
            if tablename not in stats and noOpp not in stats:
                continue
        
        # Use the extracted info to get a link with statistics for a given year
        urltext = f"https://www.teamrankings.com{linkstring}?date={date}"
        url = requests.get(urltext)
        soup = BeautifulSoup(url.text, 'html.parser')

        # Used for error handling
        table = soup.find('table')
        if table is None:
            print('Fail')
            print(date)
            print(tablename)
            continue
        else:
            tableRows = table.find_all('tr')

        
        for row in tableRows:
            
            # What "row" contains is the given statistic value for a given team in the current year, as well as other columns
            
            columns = row.find_all("td")
            if len(columns) > 0:

                # columns[1] contains the team name - occasionally this will be in link format, handle that case below
                a_tag = columns[1].find('a')
                if a_tag:
                    # Extract the text if 'a' tag is present
                    team = a_tag.text
                else:
                    # Handle the case where 'a' tag is not present
                    team = columns[1].text
                
                # Normalize team name - check QualityControl.py if interested in specifics
                team = convertTR(team)

                # This is where we find the actual value
                value = columns[2].text
                teamyear = str(team) +"_"+ str(year)
                if teamyear in teamstoskip:
                    continue
                if value == "--":
                    print(f"Null value for {teamyear} for {tablename}")
                # Add data into dictionary, by first checking if the team is already initialized
                # We seperate each team into years because there is no relation for us between 2007 and 2008 duke
                
                
                # Translate percentages in format '10.0%' into .1
                if(value[-1] == '%'):
                    if value == "--":
                        print(f"Null value for {teamyear} for {tablename}")
                        continue
                    value = percentToFloat(value)
                value = float(value)
                value = round(value, 4)
                if teamyear in teams:
                    teams[teamyear][tablename] = value
                else:
                    # If our teamyear is not included already, include it now
                    teams[teamyear] = {tablename: value}
    print(date)



# Connect to our database
connection = psycopg2.connect(
    host="localhost", port="5432", database="MarchMadness", user = "postgres", password = PASSWORD
)
crsr = connection.cursor()

# Initialize our table
crsr.execute("""CREATE TABLE IF NOT EXISTS teamrankings(
    teamyear VARCHAR(255) PRIMARY KEY NOT NULL,
    TwoPPP DECIMAL NOT NULL,
    ThreePPP DECIMAL NOT NULL,
    FtPPP DECIMAL NOT NULL,
    TwoPPerc DECIMAL NOT NULL,
    ThreePPerc DECIMAL NOT NULL,
    FTPerc DECIMAL NOT NULL,
    TrPerc DECIMAL NOT NULL,
    OrPerc DECIMAL NOT NULL,
    DrPerc DECIMAL NOT NULL,
    APP DECIMAL NOT NULL,
    NonStealTPP DECIMAL NOT NULL,
    SPP DECIMAL NOT NULL,
    BPP DECIMAL NOT NULL,
    FPP DECIMAL NOT NULL,
    oTwoPPP DECIMAL NOT NULL,
    oThreePPP DECIMAL NOT NULL,
    oTwoPPerc DECIMAL NOT NULL,
    oThreePPerc DECIMAL NOT NULL,
    oAPP DECIMAL NOT NULL,
    oNonStealTPP DECIMAL NOT NULL,
    oSPP DECIMAL NOT NULL,
    oBPP DECIMAL NOT NULL,
    oFPP DECIMAL NOT NULL,
    Pace DECIMAL NOT NULL
);""")

connection.commit()



# Mix of statistics collected for the purpose of making calculations, and others not useful for the regression
skips = ["Points per Game", "Games Played", "Steals per Game", "Turnovers per Offensive Play",
         "Opponent Points per Game", "Opponent Steals per Game", "Opponent Turnovers per Offensive Play",
         "Opponent Percent of Points from Free Throws", "Opponent Free Throw %", "Opponent Offensive Rebounding %",
         "Opponent Defensive Rebounding %",  "Opponent Total Rebounding % (Rebound Rate)",]

# Initialize our list for the executemany
overallinsert = []
for team,statsdict in teams.items():

    # Change turnover numbers into NonSteal Turnovers. Because of this we want to skip per Play numbers for turnovers, we want to use this instead
    statsdict["Turnovers per Game"] = statsdict["Turnovers per Game"] - statsdict["Opponent Steals per Game"]
    statsdict["Opponent Turnovers per Game"] = statsdict["Opponent Turnovers per Game"] - statsdict["Steals per Game"]

    # Extract the amount of plays this team(i.e. 2010 Michigan) had on offense and defense
    oPlays = numPlays(statsdict["Turnovers per Game"], statsdict["Turnovers per Offensive Play"], statsdict["Games Played"])
    dPlays = numPlays(statsdict["Steals per Game"], statsdict["Steals per Defensive Play"], statsdict["Games Played"])

    # Extract the amount of games, points per game, and opponent points per game. Needed for calculations
    games = statsdict["Games Played"]
    ppg = statsdict["Points per Game"]
    oppg = statsdict["Opponent Points per Game"]

    # Begin the row insert for the sql table
    inserts = [-11] * 25
    inserts[0] = team

    for stat, value in statsdict.items():
        # As explained earlier not needed for the regression
        if stat in skips:
            continue
        insert = value
        if (stat.endswith('per Game') and stat != "Possesions per Game")  or stat.endswith('pointers'):
            # Normalize per Game statistics into per Offensive/Defensive Play
            if stat == 'Opponent Steals per Game' or stat == "Opponent Blocks per Game" or stat == "Opponent Personal Fouls per Game":
                insert = perGameToPlay(value, games, oPlays)
            elif stat == 'Steals per Game' or stat == 'Blocks per Game' or stat == 'Personal Fouls per Game' or stat[0:8] == "Opponent":
                insert = perGameToPlay(value, games, dPlays)
            else:
                insert = perGameToPlay(value, games, oPlays)
        elif stat.endswith('Free Throws'):
            # Stat came in as Percent of Points from Free Thrwos - normalize into free throw points per play
            if(stat[0] == 'O'):
                insert = PercentFTToPlay(ppg,value,oPlays)
            else:
                insert = PercentFTToPlay(oppg, value,dPlays)
        # Because we hashed our stats they are not in order, use this function to get them back in order
        insertIndex = maptoIndex(stat)
        insert = round(insert,4)
        inserts[insertIndex] = insert
    overallinsert.append(inserts)
crsr.executemany("""
    INSERT INTO teamrankings (
        teamyear, TwoPPP, ThreePPP, FtPPP, TwoPPerc, ThreePPerc, FTPerc, TrPerc, OrPerc, DrPerc, APP, NonStealTPP, SPP, BPP, 
        FPP, oTwoPPP, oThreePPP, oTwoPPerc, oThreePPerc, oAPP, oNonStealTPP, oSPP, oBPP, oFPP, Pace
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s,  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT(teamyear) DO NOTHING
    """, overallinsert)



# Commit the transaction to save changes
connection.commit()

# Close the cursor and connection
crsr.close()
connection.close()
        


