from jsonDB import *

jql = jsonDB('ff.db')

# jql.insert({'Pigskinn_default': {},'fb_yahoo_default': {}},'game type')
# jql.insert({'Pigskinn_16': {},'Baseball_17': {}},'league')

# jql.insert({},'league','Baseball_17',"You're a Gyrok")
# jql.insert({},'league','Baseball_17',"Upper Deck")
# jql.insert({},'league','Baseball_17',"Beer4Breakfast")
# jql.insert({},'league','Baseball_17',"Josh's Team")
# jql.insert({},'league','Baseball_17',"braden's Team")
# jql.insert({},'league','Baseball_17',"arjan's Team")
# jql.insert({},'league','Baseball_17',"adam's Peerless Team")
# jql.insert({},'league','Baseball_17',"Maurice's Team")
# jql.insert({},'league','Baseball_17',"Mike's Team")
# jql.insert({},'league','Baseball_17',"Lane's Team")
# jql.insert({},'league','Baseball_17',"Cullen's Cool Team")
# jql.insert({},'league','Baseball_17',"steve's Team")

# jql.insert({},'league','Baseball_17',"You're a Gyrok",'Schedule')
# jql.insert({},'league','Baseball_17',"Upper Deck",'Schedule')
# jql.insert({},'league','Baseball_17',"Beer4Breakfast",'Schedule')
# jql.insert({},'league','Baseball_17',"Josh's Team",'Schedule')
# jql.insert({},'league','Baseball_17',"braden's Team",'Schedule')
# jql.insert({},'league','Baseball_17',"arjan's Team",'Schedule')
# jql.insert({},'league','Baseball_17',"adam's Peerless Team",'Schedule')
# jql.insert({},'league','Baseball_17',"Maurice's Team",'Schedule')
# jql.insert({},'league','Baseball_17',"Mike's Team",'Schedule')
# jql.insert({},'league','Baseball_17',"Lane's Team",'Schedule')
# jql.insert({},'league','Baseball_17',"Cullen's Cool Team",'Schedule')
# jql.insert({},'league','Baseball_17',"steve's Team",'Schedule')

# jql.insert({},'league','Baseball_17',"You're a Gyrok",'Season')
# jql.insert({},'league','Baseball_17',"Upper Deck",'Season')
# jql.insert({},'league','Baseball_17',"Beer4Breakfast",'Season')
# jql.insert({},'league','Baseball_17',"Josh's Team",'Season')
# jql.insert({},'league','Baseball_17',"braden's Team",'Season')
# jql.insert({},'league','Baseball_17',"arjan's Team",'Season')
# jql.insert({},'league','Baseball_17',"adam's Peerless Team",'Season')
# jql.insert({},'league','Baseball_17',"Maurice's Team",'Season')
# jql.insert({},'league','Baseball_17',"Mike's Team",'Season')
# jql.insert({},'league','Baseball_17',"Lane's Team",'Season')
# jql.insert({},'league','Baseball_17',"Cullen's Cool Team",'Season')
# jql.insert({},'league','Baseball_17',"steve's Team",'Season')

# import csv

# with open('BBSchedule.csv') as sch:
#     data = [a for a csv.reader(sch)]

# teams = ["Josh's Team","steve's Team","Maurice's Team","Cullen's Cool Team","adam's Peerless Team","Lane's Team","braden's Team","arjan's Team","Mike's Team","You're a Gyrok","Beer4Breakfast","Upper Deck"]

# for ndx,line in enumerate(data):
#     for num,week in enumerate(line[1:]):
#         jql.insert(week,'league','Baseball_17',teams[ndx],'Schedule','Week ' + str(num+1))

# with open('BBScores.csv') as sch:
#     data = [a for a csv.reader(sch)]

# for ndx,line in enumerate(data):
#     for num,week in enumerate(line[1:]):
#         a,b,c,d,e,f,g,h,i,j = week.split(',')
#         jql.insert({'R':a,'HR':b,'RBI':c,'SB':d,'AVG':e,'W':f,'SV':g,'SO':h,'ERA':i,'WHIP':j},'league','Baseball_17',line[1],'Season','Week ' + str(num+1))