from pydelicious import get_popular,get_userposts,get_urlposts
import random

# A dictionary of movie critics and their ratings of a small
# set of movies
from math import sqrt

#Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs,person1,person2):
    #得到sharead_items的列表
    si={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item]=1
    #如果两者没有共同之处，则返回0
    if len(si)==0:return 0
    
    #计算所有差值的平方和
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2)
                        for item in prefs[person1] if item in prefs[person2]])
    return 1/(1+sum_of_squares)

# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs,p1,p2):
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item]=1
    # Find the number of elements
    n=len(si)
    # if they are no ratings in common, return 0
    if n==0: return 0
    # Add up all the preferences
    sum1=sum([prefs[p1][it] for it in si])
    sum2=sum([prefs[p2][it] for it in si])
    # Sum up the squares
    sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
    sum2Sq=sum([pow(prefs[p2][it],2) for it in si])
    # Sum up the products
    pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
    # Calculate Pearson score
    num=pSum-(sum1*sum2/n)
    den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0: return 0
    r=num/den
    return r

# Returns the best matches for person from the prefs dictionary.
# Number of results and similarity function are optional params.
def topMatches(prefs,person,n=7,similarity=sim_pearson):
    scores=[(similarity(prefs,person,other),other)
            for other in prefs if other!=person]
    # Sort the list so the highest scores appear at the top
    scores.sort()
    scores.reverse()
    return scores[0:n]

# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(prefs,person,similarity=sim_pearson):
    totals={}
    simSums={}
    for other in prefs:
        # don't compare me to myself
        if other==person: continue
        sim=similarity(prefs,person,other)
        # ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
                
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            result[item][person]=prefs[person][item]            
    return result

def initializeUserDict(tag,count=5):
    user_dict={}
    # get the top count' popular posts
    for p1 in get_popular(tag=tag)[0:count]:
        # find all users who posted this
        for p2 in get_urlposts(p1['url']):
            user=p2['user']
            user_dict[user]={}
    return user_dict


def fillItems(user_dict):
    all_items={}
    # Find links posted by all users
    for user in user_dict:
        for i in range(3):
            try:
                posts=get_userposts(user)
                break
            except:
                print "Failed user "+user+", retrying"
#                time.sleep(4)
        for post in posts:
            url=post['url']
            user_dict[user][url]=1.0
            all_items[url]=1
    # Fill in missing items with 0
    for ratings in user_dict.values():
        for item in all_items:
            if item not in ratings:
                ratings[item]=0.0
            
#delusers=initializeUserDict('programming')  
#delusers['tsegaran']={} 
#fillItems(delusers)                              #耗时很多


#导入保存数据文件的模块
import pickle
#
#将结果存入数据文件
#with open('E:\PythonResource\PythonSolution\ProgrammingCollectiveIntelligence\ch2\delusers.txt', 'wb') as f:
    #delusers.dump(entry,f)
#

#从结果文件中读出数据

with open('E:\PythonResource\PythonSolution\ProgrammingCollectiveIntelligence\ch2\delusers.txt', 'rb') as f:   
    entry = pickle.load(f) 
#
delusers=entry
user=delusers.keys()[random.randint(0,len(delusers)-1)]
res=topMatches(delusers,user)

a=100


