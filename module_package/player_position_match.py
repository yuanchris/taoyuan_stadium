#!/usr/bin/env python
# -*- coding: UTF-8 -*- 

'''
=========================================================
File Name   :   player_position_match.py
Copyright   :   Chao-Chuan Lu
Create date :   Sep. 10, 2019
Description :   Match player with position.
=========================================================
'''

import math
import time
from itertools import permutations

def calc_player_position_match(unpairedPositions, playerWithDistances):
    """Compute the best matching between players and positions.
    
    Args:
        unpairedPositions:          [int]
        playerWithDistances:        [{"player":int,"distances":[float, ...]}, ...]
    Returns:
        matchedPlayerForPositions:  [{"position":int, "player":int}, ...]
    """
    # player distance between defender positions
    playerIdxList = range(len(playerWithDistances))
    positionCount = len(unpairedPositions)
    costList = []
    playerPositionIdxList = []
    combinations = list(permutations(playerIdxList, positionCount))
    for combi in combinations:
        costList.append(sum([playerWithDistances[playerIdx]["distances"][combi[playerIdx]] for playerIdx in range(len(playerWithDistances))]))
        playerPositionIdxList.append(combi)
    # The combination of best match index
    bestMatchIndex =playerPositionIdxList[costList.index(min(costList))]
    matchedPlayerForPositions = []
    for i in range(positionCount):
        matchedPlayerForPositions.append({"position":unpairedPositions[i], "player":playerWithDistances[bestMatchIndex[i]]["player"]})
    
    return matchedPlayerForPositions