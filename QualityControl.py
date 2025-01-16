# Calculate the amount of plays a given team had during the season - works for both defense and offense
def numPlays(pergame, perplay, games):
    totalval = pergame * games
    return totalval/perplay

# Calculate the perplay numbers using the given pergame
def perGameToPlay(pergame, games, numplays):
    total = pergame * games
    return total/numplays

def PercentFTToPlay(ppg, ftnum,numplays):
    FTPoints = ppg * ftnum
    return FTPoints/numplays

def percentToFloat(percstring):
    snipped = percstring[0:-1]
    return float(snipped) / 100

def maptoIndex(stat):
    mapper = {
        "Points from 2 pointers": 1,
        "Points from 3 pointers": 2,
        "Percent of Points from Free Throws": 3,
        "Two Point %": 4,
        "Three Point %": 5,
        "Free Throw %": 6,
        "Total Rebounding % (Rebound Rate)": 7,
        "Offensive Rebounding %": 8,
        "Defensive Rebounding %": 9,
        "Assists per Game": 10,
        "Turnovers per Game": 11,
        "Steals per Defensive Play": 12,
        "Blocks per Game": 13,
        "Personal Fouls per Game": 14,
        "Opponent Points from 2 pointers": 15,
        "Opponent Points from 3 pointers": 16,
        "Opponent Two Point %": 17,
        "Opponent Three Point %": 18,
        "Opponent Assists per Game": 19,
        "Opponent Turnovers per Game": 20,
        "Opponent Steals per Defensive Play": 21,
        "Opponent Blocks per Game": 22,
        "Opponent Personal Fouls per Game": 23,
        "Possessions per Game": 24
    }
    return mapper[stat]

# Function used to normalize team names on TeamRankings
def convertTR(team):
    if(team == "Abl Christian"):
        return "Abilene Christian"
    elif(team == "Alab A&M"):
        return "Alabama A&M"
    elif(team == "App State"):
        return "Appalachian St."
    elif(team == "Ark Pine Bl"):
        return "Arkansas Pine Bluff"
    elif(team == "Beth-Cook"):
        return "Bethune Cookman"
    elif(team == "Boston Col"):
        return "Boston College"
    elif(team == "Boston U"):
        return "Boston University"
    elif(team == "Bowling Grn"):
        return "Bowling Green"
    elif(team == "CS Bakersfld"):
        return "Cal St. Bakersfield"
    elif(team == "CS Fullerton"):
        return "Cal St. Fullerton"
    elif(team == "Cal St Nrdge"):
        return "Cal St. Northridge"
    elif(team == "Central Ark"):
        return "Central Arkansas"
    elif(team == "Central Conn"):
        return "Central Connecticut"
    elif(team == "Central Mich"):
        return "Central Michigan"
    elif(team == "Col Charlestn"):
        return "Charleston"
    elif(team == "Charl South"):
        return "Charleston Southern"
    elif(team == "Coastal Car"):
        return "Coastal Carolina"
    elif(team == "Uconn"):
        return "Connecticut"
    elif(team == "Detroit"):
        return "Detroit Mercy"
    elif(team == "E Carolina"):
        return "East Carolina"
    elif(team == "E Tenn St"):
        return "East Tennessee St."
    elif(team == "E Illinois"):
        return "Eastern Illinois"
    elif(team == "E Kentucky"):
        return "Eastern Kentucky"
    elif(team == "E Michigan"):
        return "Eastern Michigan"
    elif(team == "E Washingtn"):
        return "Eastern Washington"
    elif(team == "F Dickinson"):
        return "Fairleigh Dickinson"
    elif(team == "Florida Intl"):
        return "FIU"
    elif(team == "Fla Atlantic"):
        return "Florida Atlantic"
    elif(team == "Fla Gulf Cst"):
        return "Florida Gulf Coast"
    elif(team == "Gard-Webb"):
        return "Gardner Webb"
    elif(team == "Geo Mason"):
        return "George Mason"
    elif(team == "Geo Wshgtn"):
        return "George Washington"
    elif(team == "GA Southern"):
        return "Georgia Southern"
    elif(team == "GA Tech"):
        return "Georgia Tech"
    elif(team == "Grd Canyon"):
        return "Grand Canyon"
    elif(team == "WI-Grn Bay"):
        return "Green Bay"
    elif(team == "Hsn Christian"):
        return "Houston Christian"
    elif(team == "IL-Chicago"):
        return "Illinois Chicago"
    elif(team == "Incar Word"):
        return "Incarnate Word"
    elif(team == "IU Indy"):
        return "IUPUI"
    elif(team == "Jksnville St"):
        return "Jacksonville St."
    elif(team == "James Mad"):
        return "James Madison"
    elif(team == "AR Lit Rock"):
        return "Little Rock"
    elif(team == "Lg Beach St"):
        return "Long Beach St."
    elif(team == "UL Monroe"):
        return "Louisiana Monroe"
    elif(team == "LA Tech"):
        return "Louisiana Tech"
    elif(team == "Loyola-Chi"):
        return "Loyola Chicago"
    elif(team == "Loyola Mymt"):
        return "Loyola Marymount"
    elif(team == "Loyola-MD"):
        return "Loyola MD"
    elif(team == "Maryland ES"):
        return "Maryland Eastern Shore"
    elif(team == "U Mass"):
        return "Massachusetts"
    elif(team == "Miami"):
        return "Miami FL"
    elif(team == "Miami (OH)"):
        return "Miami OH"
    elif(team == "Middle Tenn"):
        return "Middle Tennessee"
    elif(team == "WI-Milwkee"):
        return "Milwaukee"
    elif(team == "Ole Miss"):
        return "Mississippi"
    elif(team == "Miss State"):
        return "Mississippi St."
    elif(team == "Miss Val St"):
        return "Mississippi Valley St."
    elif(team == "Mt St Marys"):
        return "Mount St. Mary's"
    elif(team == "NC State"):
        return "N.C. State"
    elif(team == "Neb Omaha"):
        return "Nebraska Omaha"
    elif(team == "N Hampshire"):
        return "New Hampshire"
    elif(team == "N Mex State"):
        return "New Mexico St."
    elif(team == "Nicholls"):
        return "Nicholls St."
    elif(team == "N Alabama"):
        return "North Alabama"
    elif(team == "N Carolina"):
        return "North Carolina"
    elif(team == "NC A&T"):
        return "North Carolina A&T"
    elif(team == "NC Central"):
        return "North Carolina Central"
    elif(team == "N Dakota St"):
        return "North Dakota St."
    elif(team == "N Florida"):
        return "North Florida"
    elif(team == "Northeastrn"):
        return "Northeastern"
    elif(team == "N Arizona"):
        return "Northern Arizona"
    elif(team == "N Colorado"):
        return "Northern Colorado"
    elif(team == "N Illinois"):
        return "Northern Illinois"
    elif(team == "N Iowa"):
        return "Northern Iowa"
    elif(team == "N Kentucky"):
        return "Northern Kentucky"
    elif(team == "NW State"):
        return "Northwestern St."
    elif(team == "U Penn"):
        return "Penn"
    elif(team == "Prairie View"):
        return "Prairie View A&M"
    elif(team == "IPFW"):
        return "Purdue Fort Wayne"
    elif(team == "Rob Morris"):
        return "Robert Morris"
    elif(team == "Sac State"):
        return "Sacramento St."
    elif(team == "Sacred Hrt"):
        return "Sacred Heart"
    elif(team == "St Josephs"):
        return "Saint Joseph's"
    elif(team == "S Alabama"):
        return "South Alabama"
    elif(team == "St Marys"):
        return "Saint Mary's"
    elif(team == "St Peters"):
        return "Saint Peter's"
    elif(team == "Sam Hous St"):
        return "Sam Houston St."
    elif(team == "SIU Edward"):
        return "SIU Edwardsville"
    elif(team == "S Florida"):
        return "South Florida"
    elif(team == "SE Missouri"):
        return "Southeast Missouri St."
    elif(team == "SE Louisiana"):
        return "Southeastern Louisiana"
    elif(team == "S Illinois"):
        return "Southern Illinois"
    elif(team == "S Indiana"):
        return "Southern Indiana"
    elif(team == "St Bonavent"):
        return "St. Bonaventure"
    elif(team == "S Mississippi"):
        return "Southern Miss"
    elif(team == "S Utah"):
        return "Southern Utah"
    elif(team == "St. Bonavent"):
        return "St. Bonaventure"
    elif(team == "St Johns"):
        return "St. John's"
    elif(team == "Ste F Austin"):
        return "Stephen F. Austin"
    elif(team == "TX Christian"):
        return "TCU"
    elif(team == "TN Martin"):
        return "Tennessee Martin"
    elif(team == "TN State"):
        return "Tennessee St."
    elif(team == "TN Tech"):
        return "Tennessee Tech"
    elif(team == "TX A&M-Com"):
        return "Texas A&M Commerce"
    elif(team == "TX A&M-CC"):
        return "Texas A&M Corpus Chris"
    elif(team == "TX Southern"):
        return "Texas Southern"
    elif(team == "Citadel"):
        return "The Citadel"
    elif(team == "UCSB"):
        return "UC Santa Barbara"
    elif(team == "UCSD"):
        return "UC San Diego"
    elif(team == "Mass Lowell"):
        return "UMass Lowell"
    elif(team == "Maryland BC"):
        return "UMBC"
    elif(team == "Kansas City"):
        return "UMKC"
    elif(team == "NC-Asheville"):
        return "UNC Asheville"
    elif(team == "NC-Grnsboro"):
        return "UNC Greensboro"
    elif(team == "NC-Wilmgton"):
        return "UNC Wilmington"
    elif(team == "SC Upstate"):
        return "USC Upstate"
    elif(team == "TX-Arlington"):
        return "UT Arlington"
    elif(team == "TX-Pan Am"):
        return "UT Rio Grande Valley"
    elif(team == "TX El Paso"):
        return "UTEP"
    elif(team == "VA Tech"):
        return "Virginia Tech"
    elif(team == "Wash State"):
        return "Washington St."
    elif(team == "W Virginia"):
        return "West Virginia"
    elif(team == "W Carolina"):
        return "Western Carolina"
    elif(team == "W Illinois"):
        return "Western Illinois"
    elif(team == "W Kentucky"):
        return "Western Kentucky"
    elif(team == "W Michigan"):
        return "Western Michigan"
    elif(team == "Wm & Mary"):
        return "William & Mary"
    elif(team == "Youngs St"):
        return "Youngstown St."
    elif(team == "St Fran (PA)"):
        return "Saint Francis"
    elif(team == "S Methodist"):
        return "SMU"
    elif(team == "S Car State"):
        return "South Carolina St."
    elif(team == "S Carolina"):
        return "South Carolina"
    elif(team == "S Dakota St"):
        return "South Dakota St."
    elif(team.endswith("St")):
        return team + "."
    else:
        return team

