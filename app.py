from flask import Flask, request, jsonify
import openai
import time
import numpy as np
import json
import math
import requests
import os
from dotenv import load_dotenv,find_dotenv

app = Flask(__name__)
#Store OpenAI API key
env_path=os.path.join('env_folder', 'credentials.env')
load_dotenv(env_path)
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
key = os.environ.get('key2')
client = openai.OpenAI(api_key= key)
# Global variables:
list_days = {'Mon':'Thứ Hai','Tue':'Thứ Ba','Wed':'Thứ Tư','Thu':'Thứ Năm','Fri':'Thứ Sáu','Sat':'Thứ Bảy','Sun':'Chủ Nhật'}

#Get Home and Away name:
def get_match_info(match_id):
    end_point_match_info = 'https://api.tysobongda.org/matchInfo'
    body = {"id":str(match_id)}
    response = requests.post(end_point_match_info,headers={"Content-Type": "application/json"},json= body)
    team1 = response.json()['result']['home_name']
    team2 = response.json()['result']['away_name']
    league_name = response.json()['result']['competition_name']
    unix_time = response.json()['result']['match_time']
    day_of_week_vi = list_days[time.strftime("%a", time.localtime(unix_time-3600))]
    date_dmy =time.strftime("%d/%m/%Y", time.localtime(unix_time-3600))
    time_hm= time.strftime("%H:%M", time.localtime(unix_time-3600))
    stadium = response.json()['result']['venue_name']
    return team1,team2,league_name,day_of_week_vi,date_dmy,time_hm,stadium

# main function
def get_match_analysis(match_id,team1,team2):
    end_point = 'https://api.tysobongda.org/matchH2H'
    body = {"id":str(match_id)}
    response = requests.post(end_point,headers={"Content-Type": "application/json"},json= body)
    data = response.json()
    # Your code to fetch match data goes here
    team1_h2h_home_scrores = []
    for i in data['h2h'][0]['matches']:
        if i['home_name']== team1:
            team1_h2h_home_scrores.append(i['home_scores'][0])
    #team1 as away
    team1_h2h_away_scrores = []
    for i in data['h2h'][0]['matches']:
        if i['away_name']== team1:
            team1_h2h_away_scrores.append(i['away_scores'][0]) 
    #team2 as home
    team2_h2h_home_scrores = []
    for i in data['h2h'][0]['matches']:
        if i['home_name']== team2:
            team2_h2h_home_scrores.append(i['home_scores'][0])
    #team2 as away
    team2_h2h_away_scrores = []
    for i in data['h2h'][0]['matches']:
        if i['away_name']== team2:
            team2_h2h_away_scrores.append(i['away_scores'][0]) 
    #Average h2h home and away scores 
    team1_avg_h2h_home_scores = np.mean(team1_h2h_home_scrores[0:5])
    team2_avg_h2h_away_scores = np.mean(team2_h2h_away_scrores[0:5])
    #Team1 GA and GF scores
    team1_home_ga_scores=[]
    team1_home_gf_scores=[]
    for i in data['home'][0]['matches']:
        if i['home_name']== team1:
            team1_home_ga_scores.append(i['home_scores'][0])
            team1_home_gf_scores.append(i['away_scores'][0])
    #H2H recents 5 games
    team1_h2h_stats = {'win':0,'draw':0,'loss':0}
    for i in data['h2h'][0]['matches'][0:5]:
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
    for i in data['home'][0]['matches'][0:5]:
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
    for i in data['away'][0]['matches'][0:5]:
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
    for i in data['home'][0]['matches']:
        if i['home_name']== team1:
            team1_home_ga_scores.append(i['home_scores'][0])
            team1_home_gf_scores.append(i['away_scores'][0])
    #Team1 adjusted EG
    team1_home_eg_adjst_ratio = np.mean(team1_home_ga_scores[0:])/np.mean(team1_home_gf_scores[0:])

    #Team2 GA and GF scores
    team2_away_ga_scores=[]
    team2_away_gf_scores=[]
    for i in data['away'][0]['matches']:
        if i['away_name']== team2:
            team2_away_ga_scores.append(i['away_scores'][0])
            team2_away_gf_scores.append(i['home_scores'][0])
    #Team2 adjusted EG
    team2_away_eg_adjst_ratio = np.mean(team2_away_ga_scores[0:])/np.mean(team2_away_gf_scores[0:])
    #Home and Away EG
    home_eg = team1_avg_h2h_home_scores * team1_home_eg_adjst_ratio
    away_eg = team2_avg_h2h_away_scores * team2_away_eg_adjst_ratio
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
    # For demonstration, we are returning dummy data
    return team1_h2h_stats,home_stats,away_stats,home_goal_pred,away_goal_pred,home_win_prob,away_win_prob,draw_prob,both_team_score_prob,home_goals_probs,away_goals_probs
def write_content4turbo(team1,team2,league_name,day_of_week_vi,date_dmy,time_hm,stadium,team1_h2h_stats,home_stats,away_stats,home_goal_pred,away_goal_pred,home_win_prob,away_win_prob,draw_prob,both_team_score_prob):
        completion = client.chat.completions.create(
        model = "gpt-4-1106-preview",
        temperature = 1.0,
        max_tokens = 2000,
        messages = [
            {"role": "system", "content": "bạn là là một chuyên gia sáng tạo nội dung cho website"}, 
            {"role": "system", "content": "bạn có am hiểu về phân tích và nhận định các trận bóng đá"}, 
            {"role": "system", "content": "Bạn chỉ đưa ra nhận định dựa trên những số liệu thống kê được cung cấp mà không sử dụng thêm thông tin bên ngoài."}, 
            {"role": "system", "content": f"""Các số liệu thống kê trước trận đấu giữa {team1} và {team2} được trình bày như sau:Trong 5 lần gặp nhau gần nhất giữa {team1} và {team2}, {team1} thắng {team1_h2h_stats['win']} thua {team1_h2h_stats['loss']} và hòa {team1_h2h_stats['draw']} 
    Trong 5 trận gần nhất của giải đấu {league_name}, {team1} thắng {home_stats['win']}, thua {home_stats['loss']} hòa {home_stats['draw']}.
    Trong 5 trận gần nhất của giải đấu {league_name} team B thắng {away_stats['win']}, hòa {away_stats['draw']}, thua {away_stats['loss']}. 
    Các chuyên gia dự đoán tỉ số của trận đấu này sẽ là {home_goal_pred}-{away_goal_pred}. Xác suất thắng của {team1} là {np.round(100*home_win_prob,2)}%, xác suất thắng của team B là {np.round(100*away_win_prob,2)}%, xác suất để 2 đội hòa nhau là {np.round(100*draw_prob,2)}% và xác suất để cả hai team cùng ghi bàn là {np.round(100*both_team_score_prob,2)}%. """}, 
            {"role": "user", "content": f"Dựa vào những thông tin trên, hãy viết một bài nhận định và dự đoán về trận đấu giữa chủ nhà {team1} và {team2} diễn ra vào {time_hm} ngày {day_of_week_vi}, {date_dmy} tại sân vận động {stadium}"},         
            {"role": "user", "content": "Bài viết phải có sắc thái chuyên nghiệp, khách quan và có tính thuyết phục"},
            {"role": "user", "content": "Bài viết phải sử dụng toàn bộ các thông tin đã được cung cấp"}]
            
        )
        return completion.choices[0].message.content
#@app.route("/")
#def hello_world():
    #return "<p>{}</p>".format(key)
@app.route('/analysis', methods=['POST'])
def match_analysis():
    data = request.json
    match_id = data['id']  # Default to empty string if 'id' not provided
    if not match_id:
        return jsonify({"error": "No match ID provided"}), 400

    if match_id:
        team1,team2,league_name,day_of_week_vi,date_dmy,time_hm,stadium = get_match_info(match_id)
        team1_h2h_stats,home_stats,away_stats,home_goal_pred,away_goal_pred,home_win_prob,away_win_prob,draw_prob,both_team_score_prob,home_goals_probs,away_goals_probs = get_match_analysis(match_id,team1,team2)
        data = write_content4turbo(team1,team2,league_name,day_of_week_vi,date_dmy,time_hm,stadium,team1_h2h_stats,home_stats,away_stats,home_goal_pred,away_goal_pred,home_win_prob,away_win_prob,draw_prob,both_team_score_prob)
        output = {
            'match_id':match_id,
            'home_name':team1,
            'away_name':team2,
            'prediction':{
                'home_win_prob':home_win_prob,
                'away_win_prob':away_win_prob,
                'draw_prob':draw_prob,
                'both_team_score_prob':both_team_score_prob,
                'home_goal_pred':home_goal_pred,
                'away_goal_pred':away_goal_pred,
                'home_goals_probs':home_goals_probs,
                'away_goals_probs':away_goals_probs
                },
            'stats':{
                'home_stats':home_stats,
                'away_stats':away_stats,
                'home_away_h2h_stats':team1_h2h_stats
            },
            'analysis':data

        }
        return json.dumps(output)
    else:
        return jsonify({"error": "Match not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)
