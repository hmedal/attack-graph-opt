'''
Created on Oct 5, 2016

@author: hm568
'''

####Define the node class#######################         
class node:
    def __init__(self, nIndex, nStatus, bLoss):
        self.nIndex = nIndex    #Node index
        self.nStatus = nStatus      #Node status
        self.bLoss = bLoss      #Breach loss
        self.child = 0      #Indicates which child node has been explored in the search algorithm
        self.goodness = 0   #Quality score indicating need for separation
        self.OA = []        #list of outgoing arcs
        self.IA = []        #list of incoming arcs
        self.attOA = []     #list of outgoing arcs in a specific attack

class arc:
    def __init__(self, aIndex, tail, head, weight, status):
        self.aIndex = aIndex
        self.tail = tail        #Tail of the arc
        self.head = head        #Head of the arc
        self.weight = weight    #Weight of the arc
        self.status = status    #Arc status
        self.securityCost = 0   #defense cost
        self.attackCost = 0     #Attack cost
        self.arcPaths = []      #List of paths that the arc belongs to
        
        
#Define the path class 
    
class path:
    def __init__(self,sourceNode,sinkNode):
        self.pathAttacks = []  #####list of attacks (attack index) in which this path is used
        self.sourceNode = sourceNode
        self.sinkNode = sinkNode
        self.removed = 0
        self.prob = 0
        
        
#Declare the paths list

Paths = []      #List of all paths
        
        
#Define the iterAttack class 

class iterAttack:
    def __init__(self, iterationIndex):
        self.attackDamage = 0
        self.attackDamageTemp = 0
        self.attackDamageTemp1 = 0
        self.scenarioLoss = []
        self.scenarioLossTemp = []
        self.scenarioLossTemp1 = []
        self.scenarioPathsIndex = dict()
        self.iterationIndex = iterationIndex
        for i in range(nScenario):
            self.scenarioPathsIndex[i] = list()       
        
#Define the attack class 

class attack:
    def __init__(self,attDamage):
        self.attDamage = attDamage
        self.attDamageTemp = attDamage
        self.attDamageTemp1 = attDamage