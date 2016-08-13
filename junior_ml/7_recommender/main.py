from recommendations import *

dis=sim_distance(critics,'Lisa Rose','Gene Seymour')
movies=transformPrefs(critics)
#Item-based
res=topMatches(movies,'Superman Returns')
#User-based
res1=getRecommendations(critics,'Toby')