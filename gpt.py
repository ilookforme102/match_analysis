from openai import OpenAI
import openai
import time
import numpy as np
import json
import math
import pymysql
import requests
import os
#

# Establish a database connection
conn = pymysql.connect(
    host='128.199.228.235', 
    user='sql_dabanhtructi', 
    password='FKb75AYJzFMJET8F', 
    database='sql_dabanhtructi',
    port = 3306
)

# Create a cursor object
cursor = conn.cursor()

# Execute a query
cursor.execute("""SELECT m.id, c.name as competition, v.name as stadium , v.city city, home_team.name home_team, away_team.name away_team, m.match_time match_time, h.h2h, h.home ,h.away, m.home_scores home_scores, m.away_scores away_scores 
FROM `wpdbtt_api_matches` as m 
LEFT JOIN `wpdbtt_api_competition` as c 
ON m.competition_id = c.competition_id 
LEFT JOIN `wpdbtt_api_venue` as v 
ON m.venue_id = v.venue_id 
LEFT JOIN `wpdbtt_api_team` as home_team 
ON m.home_team_id = home_team.team_id 
LEFT JOIN `wpdbtt_api_team` as away_team 
ON m.away_team_id = away_team.team_id 
LEFT JOIN `wpdbtt_api_match_h2h` as h 
ON m.id = h.match_id 
WHERE m.match_time >= 1702828800 AND m.match_time <1702915200 AND home_team.name = 'Backa Topola'
ORDER BY `m`.`match_time` DESC LIMIT 10
""")

# Fetch all the rows in a list of lists.
rows = cursor.fetchall()
#for row in rows:
    #print(row)

# Close the cursor and connection
cursor.close()
conn.close()
match_dict = {}
for index,row in enumerate(rows):
    item = {'match_id':row[0],
            'competition': row[1],
            'stadium':row[2],
            'city':row[3],
            'home_team':row[4],
            'away_team':row[5],
            'match_time':row[6],
            'h2h':row[7],
            'home':row[8],
            'away':row[9]
           }
    match_dict[index] = item
try:
    test = match_dict[0]
except KeyError as e:
    print(e)
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key='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',
)
def get_match_info(test):
    try:
        team1 =  test['home_team']
        stadium = test['stadium']
        team2 = test['away_team']
        league_name = test['competition']
        unix_time = test['match_time']
        list_days = {'Mon':'Thứ Hai','Tue':'Thứ Ba','Wed':'Thứ Tư','Thu':'Thứ Năm','Fri':'Thứ Sáu','Sat':'Thứ Bảy','Sun':'Chủ Nhật'}
        day_of_week_vi = list_days[time.strftime("%a", time.localtime(unix_time-0*3600))]
        date_dmy =time.strftime("%d/%m/%Y", time.localtime(unix_time-0*3600))
        time_hm= time.strftime("%H:%M", time.localtime(unix_time-0*3600))
        homeaway_h2h = json.loads(test['h2h'])[0]['matches']
        ##home and away head2head stats
        #team1 as home 
        team1_h2h_home_scores = []
        for i in homeaway_h2h:
            if i['home_name']== team1:
                team1_h2h_home_scores.append(i['home_scores'][0])
        #team2 as away
        team2_h2h_away_scores = []
        for i in homeaway_h2h:
            if i['away_name']== team2:
                team2_h2h_away_scores.append(i['away_scores'][0]) 
            
        #Avg team 1 h2h home scores 5 recent games
        team1_avg_h2h_home_scores = np.mean(team1_h2h_home_scores[0:5])
        #Avg team 2 h2h away score 5 recent games
        team2_avg_h2h_away_scores = np.mean(team2_h2h_away_scores[0:5])
        #H2H recents 5 games
        homeaway_h2h = json.loads(test['h2h'])[0]['matches'][0:5]
        team1_h2h_stats = {'win':0,'draw':0,'loss':0}
        for i in homeaway_h2h:
            if i['home_scores'] > i['away_scores']:
                if i['home_name'] == team1:
                    team1_h2h_stats['win']+=1
                else:
                    team1_h2h_stats['loss']+=1
            elif i['home_scores'] == i['away_scores']:
                team1_h2h_stats['draw']+=1
            else:
                if i['home_name'] == team1:
                    team1_h2h_stats['loss']+=1
                else:
                    team1_h2h_stats['win']+=1
        #Team1 stats 5 recent games:
        home_stats = {'win':0,'draw':0,'loss':0}
        homeaway_home  =  json.loads(test['home'])[0]['matches'][0:5]
        for i in homeaway_home:
            if i['home_scores'] > i['away_scores']:
                if i['home_name'] == team1:
                    home_stats['win']+=1
                else:
                    home_stats['loss']+=1
            elif i['home_scores'] == i['away_scores']:
                home_stats['draw']+=1
            else:
                if i['home_name'] == team1:
                    home_stats['loss']+=1
                else:
                    home_stats['win']+=1
        #Team2 stats 5 recent games
        away_stats = {'win':0,'draw':0,'loss':0}
        homeaway_away =  json.loads(test['away'])[0]['matches'][0:5]
        for i in homeaway_away:
            if i['home_scores'] > i['away_scores']:
                if i['home_name'] == team2:
                    away_stats['win']+=1
                else:
                    away_stats['loss']+=1
            elif i['home_scores'] == i['away_scores']:
                away_stats['draw']+=1
            else:
                if i['home_name'] == team2:
                    away_stats['loss']+=1
                else:
                    away_stats['win']+=1
        #Team1 GA and GF scores
        #team1_home_ga_scores=[]
        team1_home_gf_scores=[]
        homeaway_home = json.loads(test['home'])[0]['matches']
        for i in homeaway_home:
            if i['home_name']== team1:
                #team1_home_ga_scores.append(i['home_scores'][0])
                team1_home_gf_scores.append(i['away_scores'][0])
        team1_avg_home_gf_scores = np.mean(team1_home_gf_scores)
        team2_away_gf_scores=[]
        homeaway_away = json.loads(test['away'])[0]['matches']
        for i in homeaway_away:
            if i['away_name']== team2:
                team2_away_gf_scores.append(i['away_scores'][0])
                #team2_away_gf_scores.append(i['home_scores'][0])
        team2_avg_away_gf_scores = np.mean(team2_away_gf_scores)
    
        #team1_home_eg_adjst_ratio = (team1_avg_h2h_home_scores + team1_avg_home_gf_scores)/2
        #team2_away_eg_adjst_ratio = (team2_avg_h2h_away_scores + team2_avg_away_gf_scores)/2
        #Home and Away EG
        home_eg = (team1_avg_h2h_home_scores + team1_avg_home_gf_scores)/2
        away_eg = (team2_avg_h2h_away_scores + team2_avg_away_gf_scores)/2
        def poisson_goal(g,eg):
            return (math.e**(-eg)*eg**g)/(math.factorial(g))
        #Home team goals probabilty, assume goals only in range 0-7
        home_goals_probs = []
        for i in range(0,8):
            prob = poisson_goal(i,home_eg)
            home_goals_probs.append(prob)
        #Away team goals probability, assume that goals are only in range 0-7
        away_goals_probs = []
        for i in range(0,8):
            prob = poisson_goal(i,away_eg)
            away_goals_probs.append(prob)
        #Home and Away goals prediction:
        home_goal_pred = home_goals_probs.index(max(home_goals_probs))
        away_goal_pred = away_goals_probs.index(max(away_goals_probs))
        #Probability that Home team win:
        home_win_prob_list = []
        for i in range(0,len(home_goals_probs)):
            for j in range(0,len(away_goals_probs)):
                if j < i:
                    home_win_prob_list.append(home_goals_probs[i]*away_goals_probs[j])
        home_win_prob = sum(home_win_prob_list)
        #Probability that both team draw:
        draw_prob = sum(np.array(home_goals_probs)*np.array(away_goals_probs))
        #Probability that Away team win:
        away_win_prob = 1 - draw_prob - home_win_prob
        #Probability that both team will score
        both_team_score_goal_list = []
        for i in home_goals_probs:
            both_team_score_goal_list.append(i*away_goals_probs[0])
        for j in away_goals_probs[1:]:
            both_team_score_goal_list.append(j*home_goals_probs[0])
        both_team_score_prob = 1 - sum(both_team_score_goal_list)
        return team1,team2,league_name,day_of_week_vi,date_dmy,time_hm,stadium,team1_h2h_stats,home_stats,away_stats,home_goal_pred,away_goal_pred,home_win_prob,away_win_prob,draw_prob,both_team_score_prob
    except TypeError as e:
        return None

def write_content4turbo(team1,team2,league_name,day_of_week_vi,date_dmy,time_hm,stadium,team1_h2h_stats,home_stats,away_stats,home_goal_pred,away_goal_pred,home_win_prob,away_win_prob,draw_prob,both_team_score_prob):
        completion = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        temperature = 1.0,
        max_tokens = 2000,
        messages = [
            {"role": "system", "content": "bạn là là một chuyên gia sáng tạo nội dung cho website"}, 
            {"role": "system", "content": "bạn có am hiểu về phân tích và nhận định các trận bóng đá"}, 
            {"role": "system", "content": "Bạn chỉ đưa ra nhận định dựa trên những số liệu thống kê được cung cấp mà không sử dụng thêm thông tin bên ngoài."}, 
            {"role": "system", "content": f"""Các số liệu thống kê trước trận đấu giữa {team1} và {team2} được trình bày như sau:Trong 5 lần gặp nhau gần nhất giữa {team1} và {team2}, {team1} thắng {team1_h2h_stats['win']} thua {team1_h2h_stats['loss']} và hòa {team1_h2h_stats['draw']} 
    Trong 5 trận gần nhất của giải đấu {league_name}, {team1} thắng {home_stats['win']}, thua {home_stats['loss']} hòa {home_stats['draw']}.
    Trong 5 trận gần nhất của giải đấu {league_name} team B thắng {away_stats['win']}, hòa {away_stats['draw']}, thua {away_stats['loss']}. 
    AI của chúng tôi dự đoán tỉ số của trận đấu này sẽ là {home_goal_pred}-{away_goal_pred}. Xác suất thắng của {team1} là {np.round(100*home_win_prob,2)}%, xác suất thắng của team B là {np.round(100*away_win_prob,2)}%, xác suất để 2 đội hòa nhau là {np.round(100*draw_prob,2)}% và xác suất để cả hai team cùng ghi bàn là {np.round(100*both_team_score_prob,2)}%. """}, 
            {"role": "user", "content": f"Dựa vào những thông tin trên, hãy viết một bài nhận định và dự đoán về trận đấu giữa chủ nhà {team1} và {team2} diễn ra vào {time_hm} ngày {day_of_week_vi}, {date_dmy} tại sân vận động {stadium}"},         
            {"role": "user", "content": "Bài viết phải có sắc thái chuyên nghiệp, khách quan và có tính thuyết phục"},
            {"role": "user", "content": "Bài viết phải sử dụng toàn bộ các thông tin đã được cung cấp"}]
            
        )
        return completion.choices[0].message.content
team1,team2,league_name,day_of_week_vi,date_dmy,time_hm,stadium,team1_h2h_stats,home_stats,away_stats,home_goal_pred,away_goal_pred,home_win_prob,away_win_prob,draw_prob,both_team_score_prob = get_match_info(test)
start = time.time()
content = write_content4turbo(team1,team2,league_name,day_of_week_vi,date_dmy,time_hm,stadium,team1_h2h_stats,home_stats,away_stats,home_goal_pred,away_goal_pred,home_win_prob,away_win_prob,draw_prob,both_team_score_prob)
finish_time  = time.time()
print(content)
print(finish_time - start)