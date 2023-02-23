import numpy as np
from vincenty import vincenty

"""
    boston = (42.3541165, -71.0693514)
    newyork = (40.7791472, -73.9680804)

    keyphrase_scores = {
        "candidate 1" : 0.5,
        "candidate 2" : 0.8
    }

    keyphrase_coordinates = {
        "candidate 1" : [ boston, newyork ],
        "candidate 2" : [ boston ]
    }

    MoranI(keyphrase_scores , keyphrase_coordinates)
    GearyC(keyphrase_scores , keyphrase_coordinates)
    GetisOrdG(keyphrase_scores , keyphrase_coordinates)
"""

def MoranI( keyphrase_scores, keyphrase_coordinates ):
  scores = [ ]
  coordinates = [ ]
  for key, value_list in keyphrase_coordinates.items():
    for value in value_list:
      scores.append(keyphrase_scores[key])
      coordinates.append(value)
  n = len(scores)
  scores = np.array(scores)
  mean = np.mean(scores)
  adjusted_scores = [ ( score - mean ) ** 2 for score in scores ]
  moranI = n / np.sum(adjusted_scores)
  sum1 = 0.0
  sum2 = 0.0
  for i in range(n):
    for j in range(n):
      distance = 1.0 / ( np.e ** vincenty(coordinates[i], coordinates[j]) )
      sum1 += distance * ( scores[i] - mean ) * ( scores[j] - mean )
      sum2 += distance
  moranI = moranI * ( sum1 / sum2 )
  return moranI

def GearyC( keyphrase_scores, keyphrase_coordinates ):
  scores = [ ]
  coordinates = [ ]
  for key, value_list in keyphrase_coordinates.items():
    for value in value_list:
      scores.append(keyphrase_scores[key])
      coordinates.append(value)
  n = len(scores)
  scores = np.array(scores)
  mean = np.mean(scores)
  sum_adjusted_scores = np.sum( [ ( score - mean ) ** 2 for score in scores ] )
  sum1 = 0.0
  sum2 = 0.0
  for i in range(n):
    for j in range(n):
      distance = 1.0 / ( np.e ** vincenty(coordinates[i], coordinates[j]) )
      sum1 += distance * ( ( scores[i] - scores[j] ) ** 2 )
      sum2 += distance 
  gearyC = ( ( ( n - 1.0 ) * sum1 ) / ( 2.0 * sum2 * sum_adjusted_scores ) )
  return gearyC

def GetisOrdG( keyphrase_scores, keyphrase_coordinates ):
  scores = [ ]
  coordinates = [ ]
  for key, value_list in keyphrase_coordinates.items():
    for value in value_list:
      scores.append(keyphrase_scores[key])
      coordinates.append(value)
  n = len(scores)
  scores = np.array(scores)
  sum1 = 0.0
  sum2 = 0.0
  for i in range(n):
    for j in range(n):
      distance = 1.0 / ( np.e ** vincenty(coordinates[i], coordinates[j]) )
      sum1 += distance * scores[i] * scores[j]
      sum2 += scores[i] * scores[j]
  getisOrdG = sum1 / sum2
  return getisOrdG
