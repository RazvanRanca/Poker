#import sys
#sys.path.append("/afs/inf.ed.ac.uk/user/s09/s0954584/Desktop/pypoker")
#sys.path.append("/afs/inf.ed.ac.uk/user/s09/s0954584/Desktop/pypoker/.libs")
#from pokereval import PokerEval
import os

dr = "./acpc/2012/logs/2p_nolimit/"
smallBlind = 50
#pe = PokerEval()

def filterHand(h, player):
  if "STATE" not in h:
    return False, ""

  phs = h.split(":")

  hNo = int(phs[1])
  bets = phs[2].split("/")
  cards = phs[3].split("/")
  rez = phs[4].split("|")
  lastOrder=phs[5].split("|")
  firstOrder = reversed(lastOrder)

  cCards = cards[1:]
  if len(cCards) > 0:
    cCards[0] = [cCards[0][:2],cCards[0][2:4],cCards[0][4:6]]
  hCards = cards[0].split("|")
  hCards = [[hCards[0][:2],hCards[0][2:4]],[hCards[1][:2],hCards[1][2:4]]]
  curBet = [smallBlind, 2*smallBlind]
  curPlayer = 0
  nextPlayer = 1
  nbets = {}
  stages = ["pre-flop","flop","turn","river"]
  s = 0
  for bet in bets:
    curB = []
    i = 0
    while i < (len(bet)):
      if bet[i] == "f":
        curB.append({"betType":"f","betAmm":0})

      elif bet[i] == "c":
        diff = curBet[nextPlayer] - curBet[curPlayer]
        #print curPlayer, nextPlayer, diff
        curB.append({"betType":"c","betAmm":diff})
        curBet[curPlayer] += diff

      elif bet[i] == "r":
        nr = ""
        while bet[i+1] in "0123456789":
          i += 1
          nr += bet[i]
        nr = int(nr)

        diff = nr - curBet[curPlayer] - (curBet[nextPlayer] - curBet[curPlayer])
        curB.append({"betType":"r","betAmm":diff})
        curBet[curPlayer] += diff + (curBet[nextPlayer] - curBet[curPlayer])
      curPlayer = nextPlayer
      nextPlayer = (nextPlayer+1)%2
      i += 1
    nbets[stages[s]] = curB
    s += 1
  bets = nbets
  
  if len(cCards) > 0 and cCards[0][0][1] == cCards[0][1][1] and cCards[0][1][1] == cCards[0][2][1]:
    suit = cCards[0][0][1]
    if "flop" not in bets or len(bets["flop"]) == 0:
      return False, ""
    if bets["flop"][0]["betType"] == "r" and lastOrder[0] == player:
      if hCards[0][0][1] != suit and hCards[0][1][1] != suit:
        return True, "1" # bluff
      if hCards[0][0][1] == suit and hCards[0][1][1] == suit:
        return True, "0" # valid bet
    if bets["flop"][0]["betType"] == "c" and bets["flop"][1]["betType"] == "r" and lastOrder[1] == player:
      if hCards[1][0][1] != suit and hCards[1][1][1] != suit:
        return True, "1" #bluff
      if hCards[1][0][1] == suit and hCards[1][1][1] == suit:
        return True, "0" #valid bet

  return False, ""

def processFile(f, player):
  gh = []
  with open(f,'r') as inp:
    for line in inp:
      rez, tag = filterHand(line[:-1], player)
      if rez:
        gh.append(line[:-1] + " " + tag)
  return gh

if __name__ == "__main__":
  fdir = "processed/"
  players = ["azure_sky","dcubot","hugh","hyperborean","little_rock","spewy_louie","neo_poker_lab","tartanian5","sartre","lucky7_12","uni_mb_poker"]
  for player1 in players:
    for player2 in players:
      if player1 == player2:
        continue
      for curPlayer in [player1, player2]:
        fs = (filter(lambda x: player1 in x and player2 in x, os.listdir(dr)))
        print player1, player2, len(fs)
        if len(fs) == 0:
          continue
        fn = fdir + curPlayer + "." + player1 + "-" + player2
        try:
           with open(fn):
            print fn, "already exists"
        except IOError:
          hands = []
          count = 0
          for f in fs:
            if count % 10 == 0:
              print count/10,
            count += 1
            hands += processFile(dr + f, curPlayer)

          bluffs = len(filter(lambda x: x[-1]=="1", hands))
          print fn, len(hands), bluffs
          with open(fn, 'w') as f:
            f.write('\n'.join(hands))
      
