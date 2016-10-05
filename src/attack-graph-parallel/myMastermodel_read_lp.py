'''
Created on Oct 5, 2016

@author: tb2038
'''

'''
Created on Aug 25, 2016

@author: tb2038
'''

#from __future__ import division  #for avoiding division problem with float values in python 2.7
import random
import datetime
from gurobipy import *
#from inspect import BoundArguments
import copy
import math
import numpy as np

#nSample = 1
nScenario= 4    #No of Scenarios
tolerance= 0.05 #gap between the upper and lower bound
SD = 0.05       #standard deviation
Bd= 3           #Defender's budget
Ba= []  #Attacker's budget
Gap = 80
###Define the risk parameters
Lamda = 0.1
Alpha = 0.9
Value_at_risk = 90 

##Control parameters for enhancements of CCG algorithm
Hi = 1      #Heuristic for initial master problem solution
Ms = 1      #Add multiple sub problem solution to the master problem
TR = 0      #Add trust region cut to the master problem
Hf = 1      #Apply the final heuristic

####################################################
#subModel= Model("subproblem")    
#### Build the code for Constraint and Column generation algorithm
#############################################################################################


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
        
#Declare the iterattacks list
IterAttacks = []        
        
#Define the attack class 

class attack:
    def __init__(self,attDamage):
        self.attDamage = attDamage
        self.attDamageTemp = attDamage
        self.attDamageTemp1 = attDamage
        
#Declare the attacks list

Attacks = []

nNodes = 0
nArcs = 0
arcs = 0        #arcs counter
N =  []      #List of all nodes
NS = []     #List of goal nodes
NI = []     #List of initially vulnerable nodes
NR = []     #List of transition nodes
A = []      #List of all arcs
counter = 0
UB = 0

###Declare the function for solving master problem
#######################################################################
def calcMasterproblem (myMasterModel, theta, x, m):
    global Hi
    '''
    if m == 0:
        ###Add theta variable
        theta = myMasterModel.addVar(lb=0, ub=GRB.INFINITY, obj=1, name='theta')
        ##Define the interdiction variable
        x = {}
        for a in range(len(A)):
            x[a] = myMasterModel.addVar(vtype=GRB.BINARY, name='x'+str(A[a].tail)+str(A[a].head))
            ##Update the model
        myMasterModel.update()
        myMasterModel.addConstr(quicksum(A[a].securityCost* x[a] for a in range(len(A))) <= Bd)
        m = m + 1
    '''    
    myMasterModel.modelSense= GRB.MINIMIZE
    #masterModel.write("masterModel.lp")
    myMasterModel.setParam("OutputFlag", 0)
  
    
    myMasterModel.optimize() 
    
    ########declare the dictionary of the interdiction values
    xval={}
    xval= myMasterModel.getAttr('x', x)
    
    etaVal = {}
    etaVal=myMasterModel.getAttr('x', eta)
    
    vVal = {}
    vVal = myMasterModel.getAttr('x', v)
    
    uVal = {}
    uVal = myMasterModel.getAttr('x', u)
    
##########################################################################  
###################### Heuristic for initial solution - select arcs based on the calculated metric ######
    if Hi==1:
    #Set xVal to 1 for some arcs based on the goodness of the goal nodes
    #print("here..................")
        NSsorted = sorted(NS,key=lambda x: x.goodness, reverse=True)
        tSecCost = 0
        for i in range(len(NSsorted)):
            for j in range(len(NSsorted[i].IA)):
                tSecCost = tSecCost+A[NSsorted[i].IA[j]].securityCost
                #print("tSecCost",tSecCost)
                if tSecCost <= Bd:
                    xval[NSsorted[i].IA[j]]=1
                    #print("xVal",NSsorted[i].IA[j],xVal[NSsorted[i].IA[j]])
                else:
                    #print("Budget exceeded")
                    break
            if tSecCost > Bd:
                break
        Hi = 0
    
    
    return xval, myMasterModel.objVal,theta, x, m, etaVal, vVal, uVal 
    
#Declare the function for solving subproblem
##########################################################################
def calcForEachScenario(n,X):
    subModel= Model("subproblem")
    
    z = {}

    for t in range(len(N)):
        z[t]= subModel.addVar(obj= N[t].bLoss, name= 'z'+str(N[t].nIndex))
    
    w = {}
    for a in range(nArcs):
        w[a] = subModel.addVar(vtype = GRB.BINARY, name = 'w'+str(A[a].tail)+ str(A[a].head))
    y = {}
    for a in range(nArcs):
        y[a] = subModel.addVar(lb = 0, ub= 1, name = 'y' + str(A[a].tail)+ str(A[a].head))
    
    subModel.modelSense= GRB.MAXIMIZE
    subModel.update()
    
    ###Constraints 3 (b)
    
    subModel.addConstr(quicksum(A[a].attackCost * w[a] for a in range(len(A)))<= Ba[n], name = "3b" )
    ###Constraints 3 (c)
    interdict = {}
    for a in range(len(A)):
        interdict[a] = subModel.addConstr(w[a]<= 1-X[a], name = "3c"+str(A[a].tail)+ str(A[a].head))
    ###Constraints 3 (d)
    MaxProb = {}
    for j in range(len(Nj)):
        MaxProb[Nj[j].nIndex] = subModel.addConstr(z[Nj[j].nIndex]<= quicksum(MeanProb[b] * y[b] for b in Nj[j].IA), name = "3d"+str(Nj[j].nIndex))
    
    ###Constraints 3(e)
    linearize1 = {}
    for a in range(len(A)):
        linearize1[a] = subModel.addConstr(y[a]>= z[A[a].tail] - (1-w[a]), name = "3e"+str(A[a].tail)+ str(A[a].head))
    ### Constraints 3(f)
    linearize2 = {}
    for a in range(len(A)):
        linearize2[a] = subModel.addConstr(y[a]<= w[a], name = "3f"+str(A[a].tail)+ str(A[a].head))
       
    ### Constraints 3(g)
    linearize3 = {}
    for a in range(len(A)):
        linearize3[a] = subModel.addConstr(y[a] <= z[A[a].tail], name = "3g"+str(A[a].tail)+ str(A[a].head) )
    
    ### Constraints 3(h)
    Arc_usageC = {}
    for j in range(len(Nj)):
        Arc_usageC[j] = subModel.addConstr(quicksum(w[a] for a in Nj[j].IA)<=1, name = "3h"+str(j))

    subModel.setParam("OutputFlag", 0)
    subModel.update()
    subModel.write("subModel.lp")
    subModel.optimize() 
    return subModel, subModel.objVal, subModel.getAttr('x', w), subModel.getAttr('x', z), subModel.SolCount,w
    #return subModel.objVal, subModel.SolCount
    
###############################END of subproblem formulation#############
########################################################################        


##Read the Input file ######################################################
###########################################################################
filename="TInput1.txt"
#filename= "TInput1.txt, Tnet25-0t5, Tnet50-0t5, Tnet100-3t5"
iFile = open(filename,'r')
data = iFile.readlines()
for vline in data:          #Read each line 
    vrow = vline.split()    #Create a list containing the elements of the row of the data file
    if len(vrow) != 0:      #Read only line with numbers
        if counter == 0:        #Number of nodes and number of arcs on the first line
            nNodes = int(vrow[0])
            nArcs = int(vrow[1])
        elif counter <= nNodes:     #Read the nodes            
            N.append(node(int(vrow[0]), int(vrow[1]), float(vrow[2])))       
            if int(vrow[1]) == 2:       #Initially vulnerable nodes
                NI.append(N[-1])        # Access the node just created 
            elif float(vrow[2]) > 0:        #Goal nodes
                NS.append(N[-1])
                UB = UB +float(vrow[2])
            else:                           #Transition nodes
                NR.append(N[-1])
        else:       #Read the arcs
            if N[int(vrow[1])].nStatus != 2:        #Read only arcs that have non-initially vulnerable head
                A.append(arc(arcs, int(vrow[0]), int(vrow[1]), float(vrow[2]), float(vrow[3])))
                A[-1].securityCost = float(vrow[4])
                A[-1].attackCost = float(vrow[5])
                N[int(vrow[0])].OA.append(arcs)
                N[int(vrow[1])].IA.append(arcs)
                arcs = arcs+1
        counter = counter+1
        
###################### Heuristic for initial solution - calculate the metric ############################
if Hi==1:
    #Determine the goodness of the goal nodes
    maxSecCost = max(a.securityCost for a in A)
    maxAttCost = max(a.attackCost for a in A)
    #print("attackCost",maxAttCost)
    maxbLoss = max(n.bLoss for n in NS)
    #print("maxbLoss",maxbLoss)
    maxOA = max(len(n.OA) for n in NS)  
    #print("maxOA",maxOA)
    for i in range(len(NS)):
        iInterdictCost = sum(A[a].securityCost/maxSecCost for a in NS[i].IA)
        oAttackCost = sum(A[a].attackCost/maxAttCost for a in NS[i].OA)
        stdbLoss = NS[i].bLoss/maxbLoss
        if len(NS[i].OA) == 0:
            NS[i].goodness = stdbLoss/(iInterdictCost+1)
        else:
            oAttackCost = oAttackCost/len(NS[i].OA)
            d = 1+.35*(len(NS[i].OA)/maxOA)
            NS[i].goodness = (d*stdbLoss)/(iInterdictCost+oAttackCost)
        #print("iInterdictCost",iInterdictCost)
        #print("oAttackCost",oAttackCost)
        #print("goodness",NS[i].nIndex,NS[i].goodness)
########################### End of code block for reading file #########################################
        
#####Calculate the set of nodes N\NI
Nj = [s for s in N if s.nStatus != 2]
#for j in Nj:
    #print('Nj:', j.nStatus)
    
#Generate the random attacker budgets
for i in range(nScenario):
    Ba.append(random.uniform(120, 160))
print('attacker budget:', Ba)
  
###Generate the probabilities of the scenarios
Scenario_Prob = []


Rndprob = []
for i in range(nScenario):
    Rndprob.append(random.uniform(0, 1))
sum_Rndprob = sum(Rndprob)
for i in range(nScenario):
    Scenario_Prob.append(Rndprob[i]/sum_Rndprob)
print("Probability of the scenarios",Scenario_Prob)


MeanProb=[]


for m in range(nArcs):
    MeanProb.append(random.uniform(0,1))
print("Mean Probabilities of attack success on arcs", MeanProb)


    
Iteration = 0
#while Gap > tolerance:
    #print('Iteration number:', Iteration)

    
    ####declare the u variables
u = {}

    ##############declare the v variables

v = {}

    ##############declare the eta variables

eta = {}

bta = {}
    
X={} 
theta = 0
x={}
m=0    
LB = 0    


CumArcs = set() #Set of arcs added so far in the model
AllAttacks = [] #List of all attacks (list of all arcs)
AllAttPaths = [] #List of paths used in all the attacks
allInterdicts = [] #All the interdiction plans as a set of arcs

masOptObjVal = {}
masHeuObjVal = {}
iterCount = 0


masStableConst = 0      #Master problem stabilization constraint (Doesn't allow big jump in the solution)
if TR==1:           #Add trust region cut up to TRiter number of iterations
    TRiter = 20   #No. of iterations upto which the mastermodel solution stabilizing constraints is active
    masMaxJump = 0.33        #Maximum jump in the master problem solution
else:
    TRiter = 0

if Ms==1:                   #Add multiple sub problem solutions to the master problem
    fracSols = 0.33         #Fraction of all sub problem solutions added to the master problem (best fracSols = 0.33)
else:
    fracSols = 0


##########################################################################  

##########################################################################    
#Define the function to Select a set of arcs to interdict using heuristics ##################### 

def arcSelect(bestArcs):
    interdictPaths = []
    tCost = 0       #Keep track of the total interdiction cost
    Allarcs = copy.copy(bestArcs)   
    while tCost<Bd and max(IA.attackDamageTemp1 for IA in IterAttacks)>0:    #Select a new arc until budget is exhausted
        maxDamage = max(IA.attackDamageTemp1 for IA in IterAttacks)        #Current maximum damage after interdiction of some arcs
        maxScore = 0
        maxArc = 10000
        for j in Allarcs:         #Evaluate all the arcs by calculating the score
            if tCost+A[j].securityCost <= Bd:
                #print("Evaluated arc",j)
                #print("arcPaths",A[j].arcPaths) 
                for p in A[j].arcPaths:         #Reduce the damage from all the paths in which arc j is used
                    if (p in bestPathsIndices) and  Paths[p].removed == 0:     #Reduce the damage if the path has not been removed already
                        for itr in IterAttacks:
                            for scl in range(len(itr.scenarioLoss)):
                                if p in itr.scenarioPathsIndex[scl]:
                                    itr.scenarioLossTemp[scl] = itr.scenarioLossTemp[scl] - N[Paths[p].sinkNode].bLoss*Paths[p].prob
                                else:
                                    itr.scenarioLossTemp[scl] = itr.scenarioLossTemp[scl]
                                    
                for itr in IterAttacks:
                    Scenario_Loss = []
                    for scl in range(len(itr.scenarioLoss)):
                        Scenario_Loss.append(itr.scenarioLossTemp[scl])
                    Sorted_Scenario_Loss = sorted(Scenario_Loss)
                    #print ("Sorted subproblem solution:", Sorted_SubSolutionLst)
                    ############################################################################
                    ##Calculate the nth percentile of the Subproblem obj values
                    LossArray = np.array(Sorted_Scenario_Loss)
                    VaR = np.percentile(LossArray, Value_at_risk)  ##Returns n th percentile
                    
                    #print("valueAtRisk:", valueAtRisk)
                    ##########################################################################
                    ##Calculate the extra loss amount
                    Extra_Loss = [] 
                    for scl in range(len(Scenario_Loss)):
                        Subtract = Scenario_Loss[scl]- VaR
                        if Subtract > 0:
                            Extra_Loss.append(Subtract)
                        else:
                            Extra_Loss.append(0)
                    ###Calculate the risk averse subproblem obj value
                    Loss_CVaR = 0
                    sumProduct1 = 0
                    sumProduct2 = 0
                    for scl in range(len(Scenario_Loss)):
                        sumProduct1 = sumProduct1 + (Scenario_Prob[scl] * Scenario_Loss[scl])
                        sumProduct2 = sumProduct2 + (Scenario_Prob[s] * Extra_Loss[s])
                        
                    Loss_CVaR = sumProduct1 + (Lamda*(VaR + 1/(1-Alpha)*sumProduct2))
                    itr.attackDamageTemp = Loss_CVaR
                    ##Reset the scenarioLossTemp list
                    for scl in range(len(itr.scenarioLoss)):
                        itr.scenarioLossTemp[scl] = itr.scenarioLossTemp1[scl]
                
                maxDamageTemp = max(IA.attackDamageTemp for IA in IterAttacks)
                Saved = maxDamage - maxDamageTemp
                Score = Saved/A[j].securityCost
                
                if Score>maxScore:
                    maxScore = Score
                    maxArc = j
                for itr in IterAttacks:
                    itr.attackDamageTemp = itr.attackDamageTemp1
        
        #print("maxArc",maxArc)       
        if maxScore>0:
            for p in A[maxArc].arcPaths:
                if (p in bestPathsIndices) and  Paths[p].removed == 0: 
                    Paths[p].removed = 1
                    u[p].start = 1      #Set the path removal variable to 1
                    interdictPaths.append(p)
                    
                    for itr in IterAttacks:
                        for scl in range(len(itr.scenarioLoss)):
                            if p in itr.scenarioPathsIndex[scl]:
                                itr.scenarioLossTemp1[scl] = itr.scenarioLossTemp1[scl] - N[Paths[p].sinkNode].bLoss*Paths[p].prob
                            else:
                                itr.scenarioLossTemp1[scl] = itr.scenarioLossTemp1[scl]
                
            for itr in IterAttacks:
                Scenario_Loss = []
                for scl in range(len(itr.scenarioLoss)):
                    Scenario_Loss.append(itr.scenarioLossTemp1[scl])
                Sorted_Scenario_Loss = sorted(Scenario_Loss)
                LossArray = np.array(Sorted_Scenario_Loss)
                VaR = np.percentile(LossArray, Value_at_risk)
                
                Extra_Loss = [] 
                for scl in range(len(Scenario_Loss)):
                    Subtract = Scenario_Loss[scl]- VaR
                    if Subtract > 0:
                        Extra_Loss.append(Subtract)
                    else:
                        Extra_Loss.append(0)
                
                Loss_CVaR = 0
                sumProduct1 = 0
                sumProduct2 = 0
                for scl in range(len(Scenario_Loss)):
                    sumProduct1 = sumProduct1 + (Scenario_Prob[scl] * Scenario_Loss[scl])
                    sumProduct2 = sumProduct2 + (Scenario_Prob[s] * Extra_Loss[s])
                    
                Loss_CVaR = sumProduct1 + (Lamda*(VaR + 1/(1-Alpha)*sumProduct2))
                itr.attackDamageTemp1 = Loss_CVaR
                itr.attackDamageTemp = itr.attackDamageTemp1
                
                for scl in range(len(itr.scenarioLoss)):
                    itr.scenarioLossTemp[scl] = itr.scenarioLossTemp1[scl]
                
            tCost = tCost+A[maxArc].securityCost
            Allarcs.remove(maxArc)
        
        else:   ##NO MORE SAVINGS POSSIBLE BY INTERDICTING ARC
            break
    
    #Store the master problem objective value from the heuristic
    maxDamage = max(IA.attackDamageTemp1 for IA in IterAttacks)
    #print("maxDamage",maxDamage)
    masHeuObjVal[iterCount] = maxDamage
    
    #Reset the paths and attack attributes for the next iteration of the CCG algorithm
    #print("interdictPaths",interdictPaths)
    for p in range(len(Paths)):
        Paths[p].removed = 0
        if k not in interdictPaths:
            u[p].start = 0
    
    for itr in IterAttacks:
        itr.attackDamageTemp1 = itr.attackDamage
        itr.attackDamageTemp = itr.attackDamage
        for scl in range(len(itr.scenarioLoss)):
            itr.scenarioLossTemp[scl] = itr.scenarioLoss[scl]
            itr.scenarioLossTemp1[scl] = itr.scenarioLoss[scl]
######################################################################################################################
##End of arc select function##########################################################################################       

                                        
bestArcs = list()       
bestPathsIndices = list()               
saveIndex = -1

k = 0 # iteration counter


# create the master model and write it to masterModel.lp
masterModel = Model("master")
theta = masterModel.addVar(lb=0, ub=GRB.INFINITY, obj=1, name='theta')
x = {}
for a in range(len(A)):
    x[a] = masterModel.addVar(vtype=GRB.BINARY, name='x'+str(A[a].tail)+str(A[a].head))

masterModel.update()
masterModel.addConstr(quicksum(A[a].securityCost* x[a] for a in range(len(A))) <= Bd)
masterModel.update()
masterModel.modelSense= GRB.MINIMIZE
masterModel.write("masterModel.lp")


##############Start of CCG Algorithm##############################################################
##################################################################################################
t1 = datetime.datetime.now()    #Start time of the algorithm
masTime = 0        #Total runtime of the master problem
masTimeAll = []     #Master problem runtime in all the iterations
masVarAll = []      #Master problem number of variables in all iterations
masConstAll = []    #Master problem number of constraints in all iterations
subTime = 0         #Total runtime of the sub problem
algTime = 0         #Total time spent so far
Num_of_uVar = 0
Start_index = 0
Last_index = 0

while UB-LB >= tolerance * UB:
    
    ####Beginning of Solving the master problem loop###############################################
    mast1 = datetime.datetime.now()
    myMasterModel = read("masterModel.lp")
    
    Vars = myMasterModel.getVars()
    print "Variables", Vars
    varMap = {}
    for var in Vars:
        varMap[var.VARNAME] = var
    print Vars
    print varMap
    print varMap['x02']
    theta = Vars[0]
    etaRead = {}
    for index in range(0,k+1):
        etaRead[index] = varMap['eta'+str(k)]
    Var_x = Vars[1: nArcs+1]
    for a in range(len(A)):
        x[a] = Var_x[a]
    
    '''     
    if k!=0:
        Start_index = nArcs+1
        Last_index = Start_index+k+1
        Var_eta = Vars[Start_index : Last_index]
        for k in range(k):
            eta[k] = Var_eta[k]
        
        Start_index = Last_index
        Last_index = Last_index + 1 +k*nScenario 
        Var_v = Vars[Start_index: Last_index]
        for k in range(k):
            for s in range(nScenario):
                v[k, s] = Var_v[k, s]
        
        Start_index = Last_index
        Last_index = Last_index + 1 +k*nScenario 
        Var_bta = Vars[Start_index: Last_index]
        for k in range(k):
            for s in range(nScenario):
                bta[k, s] = Var_bta[k, s]
        
        Start_index = Last_index
        Last_index = Last_index + 1 + Num_of_uVar
        Var_U = Vars[Start_index: Last_index]
        for p in range(Num_of_uVar):
            u[p] = Var_U[p] 
    '''
            
    X, LB_temp , theta, x,m, etaVal, vVal, uVal = calcMasterproblem(myMasterModel, theta, x, m)
    ###Calculating the time to solve the master problem and the number of variables and constraints in the myMasterModel in this iterations
    masTime = masTime+(datetime.datetime.now() - mast1).total_seconds()
    masTimeAll.append((datetime.datetime.now() - mast1).total_seconds())
    masVarAll.append(myMasterModel.NumVars)
    masConstAll.append(myMasterModel.NumConstrs)
    
    print("interdiction plan:", X)
    
    if iterCount<TRiter:
        LB = 0.0
    else:
        LB = LB_temp     
    
    print("Lower Bound:", LB)
    
    
    ################################################################
    ###Starting  of scenario loop
    SubSolutionLst = []  ##List of the subproblem objective function values from different scenarios
    LossExprLst = []    ##List of expected loss expressions from all attack plans(scenarios)
    #Add the eta variable for the current iteration
    eta[k] = myMasterModel.addVar(lb = 0, ub = GRB.INFINITY, name = "eta"+str(k)) 
    myMasterModel.update()
       
    for s in range(nScenario): 
        #print("SCENARIO: ",s)
        #bestInserted = False
        subt1 = datetime.datetime.now() 
        subModel, subobj, wVal, zVal, solCount, w = calcForEachScenario(s, X)
        
        subTime = subTime+(datetime.datetime.now()-subt1).total_seconds()
        
        #print("Subobj", subobj)
        SubSolutionLst.append(subobj) ##Stores the subproblem objective values in the list
        #print("list of subsolutions", SubSolutionLst)
        ###Taking multiple subproblem solution for each scenario
    
        #solCount = subModel.SolCount
        
        #numSols = solCount
        numSols = round(fracSols*solCount)         
        if numSols == 0:
            numSols = 1
        #print("SolCount",iterCount, s, numSols)                
        ##Add the excess variable v for the current scenario
        v[k,s] = myMasterModel.addVar(lb=0,name="v"+str(k)+str(s))    

        ##Add the beta variable for each scenario#############
        bta[k, s] = myMasterModel.addVar(lb =0, name = "bta"+str(k)+str(s))
        myMasterModel.update() 
        
        ##Beginning of the sub-problem solution loop
        subModel.params.outputFlag = 0
        for sol in range(int(0), int(numSols)):
            #print("SOLUTION: ",j)
            #Find the arcs and nodes in the attack plan of the sub problem solution
            
            attNodes = []   #Nodes used in this attack
            attArcs = []    #Arcs used in this attack
            attNI = []      #Initially vulnerable nodes used in this attack
            attNG = []      #Goal nodes used in this attack
            newArcs = set()     #New arcs used in this attack
            
            #print(" numSols", numSols)
            subModel.params.SolutionNumber = sol
            
            #wVal = {} #value of w variable which indicates the use of an arc in this attack (scenario)
            #zVal = {}  #value of z variable which indicate the probability of breaching a node in this scenario
            #wVal = subModel.getAttr('xn', w)
            #print("arcs used in attack", wVal)
    
            ##Find out the nodes and arcs used in attack using w values
            
            wVal = subModel.getAttr('Xn', w)
            #print("----------------wVal: ",j,  wVal)
            
            for a in range(len(A)):
                if wVal[a] >= 0.99:
                    attArcs.append(A[a].aIndex)
                    if N[A[a].tail].nStatus == 2:
                        attNodes.append(A[a].tail)
                        attNI.append(A[a].tail)
                    attNodes.append(A[a].head)
                    N[A[a].tail].attOA.append(A[a].aIndex)
                    if N[A[a].head].nStatus == 1:
                        attNG.append(A[a].head)
        
                  

            # we use set to remove the duplicate items             
            attNodes = set(attNodes)        #Remove the duplicate nodes
            attNI = set(attNI)
            attNG = set(attNG)
            attArcs = set(attArcs)      #Remove the duplicate arcs
        
            if sol == 0:
                for a in attArcs:
                    bestArcs.append(a)  
                bestArcs = set(bestArcs)
                bestArcs = list(bestArcs)
            
            ##Find out and append the new arcs (in this attack) to the cumulative list of all arcs (in all the iterations upto this)
            #newArcs = attArcs-CumArcs      #Find the new arcs
            #CumArcs = list(CumArcs)
            #newArcs = list(newArcs)
            #for a in newArcs:
                #CumArcs.append(a)
            #CumArcs = set(CumArcs)
            #print("New arcs",newArcs)
            #print("All arcs",CumArcs)
    
            #Determine if this attack is a new attack
            if attArcs in AllAttacks:
                print("")
                #print("Old attack repeated")
            elif len(attArcs)!=0:
                #print("New attack")
                AllAttacks.append(attArcs)
            #else:
                #print("")
                #print("No attack")
            
        
            #attDamage = 0 ##Expected loss from this attack      
            
            #Find all the distinct paths used in this attack plan using a search algorithm (depth first search)
           
            attPath = []    #List of attack paths used in this attack
            LossExpr = 0    #Expected loss from an attack plan (solution of each scenario)
            for j in attNI:         #Explore the trees of all the initially vulnerable nodes
                tempNode = j
                treeNotExplored = True
                while treeNotExplored:      #Explore the tree of an initially vulnerable node to find all the paths
                    #print("...While....")
                    if N[tempNode].child < len(N[tempNode].attOA):  # if the index of child of that node is less than the number of outgoing arcs from that node used in this attack 
                        attPath.append(N[tempNode].attOA[N[tempNode].child])
                        N[tempNode].child = N[tempNode].child+1    #Go to the next child which is another branch
                        #print("There's still child left...")
                        if N[A[attPath[-1]].head].nStatus == 1:    #Check if a goal node (leaf) is found 
                            #print("Path found")
                            #print("Recent path", attPath[-1])
                            attPathTemp = copy.copy(attPath)
                            #print("attPathTemp", attPathTemp)
                            ###Calculate the path probability 
                            PathProb = 1.0  
                            for a in attPathTemp: ##takes values from the list attPathTemp
                                PathProb = PathProb * MeanProb[a]
                            ####End of path probability calculation
                            ############################################
                            ##Attack damage calculation###########################
                            #attDamage = attDamage + N[A[attPath[-1]].head].bLoss*PathProb 
                            #######################################################
                            
                            if attPathTemp in AllAttPaths:
                                #print("Path already exists")
                                ###Generate the expected loss expression for this path and add this to the cumulative loss expression for this scenario
                                LossExpr = LossExpr + N[A[attPath[-1]].head].bLoss*PathProb *(1-u[AllAttPaths.index(attPathTemp)])
                                ###Add the new attack index to the path attribute pathAttacks (Existing path is used in this attack) 
                                Paths[AllAttPaths.index(attPathTemp)].pathAttacks.append(len(Attacks)-1)
                                #Paths[AllAttPaths.index(attPathTemp)].prob = PathProb
                                saveIndex = AllAttPaths.index(attPathTemp)
                                #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Path already exists: ", saveIndex)
                            else:
                                ##Add a path object with source node and sink node attribute to the Paths list for the current attack path used
                                Paths.append(path(A[attPathTemp[0]].tail,A[attPathTemp[-1]].head))
                                saveIndex = len(Paths)-1
                                #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Appending path: ", saveIndex)
                                ##Add the attack index to the path attribute pathAttacks of the current path used
                                Paths[-1].pathAttacks.append(len(Attacks)-1)
                                Paths[-1].prob = PathProb
                                ##add the path to the arcPath attribute of those arcs that comprises this path
                                for acs in attPathTemp:
                                    A[acs].arcPaths.append(len(Paths)-1)
                                                               
                                AllAttPaths.append(attPathTemp)
                                #print("all attack path", AllAttPaths)
                                u[len(AllAttPaths)-1] = myMasterModel.addVar(lb=0,ub=1,name="u"+str(len(AllAttPaths)-1))
                                myMasterModel.update()
                                Num_of_uVar = Num_of_uVar + 1
                                
                                LossExpr = LossExpr + N[A[attPath[-1]].head].bLoss*PathProb*(1-u[len(AllAttPaths)-1])
                                #Add constraint (5c) to the myMasterModel
                                myMasterModel.addConstr(u[len(AllAttPaths)-1] <= quicksum(x[l] for l in attPathTemp),"5c"+str(len(AllAttPaths)-1))
                                                                             
                        
                        tempNode = A[attPath[-1]].head
                        
                        #else:
                            #print("Path not found")
                    else:
                        tempNode = A[attPath[-1]].tail
                        attPath.pop()
                        #print("Tempnode updating2: ",tempNode)
                    
                    if len(attPath)==0 and N[j].child>=len(N[j].attOA):
                        treeNotExplored = False
                        #print("N[j].child: ",j, N[j].child,N[j].attOA)
                        #print("Tree explored")
                       
                    if sol == 0 and saveIndex>-1:
                        bestPathsIndices.append(saveIndex)         
                        bestPathsIndices = set(bestPathsIndices)
                        bestPathsIndices = list(bestPathsIndices)
                        if len(IterAttacks) == k:#New entry for this iteration
                            IterAttacks.append(iterAttack(k)) ##Create the iterattack object with the index k
                            #ita = iterAttack(k)
                            IterAttacks[k].scenarioPathsIndex[s] = [saveIndex] ##Creating the scenario index as the key to the dictionary of scenarioPathsIndex  
                            #IterAttacks.append(ita)
                        else: #Iteration exist
                            if s in IterAttacks[k].scenarioPathsIndex:  ##if the key s exists (already created previously)
                                IterAttacks[k].scenarioPathsIndex[s].append(saveIndex) ##Append the index of another path to that list of path in the scenario with key s                       
                            else:
                                IterAttacks[k].scenarioPathsIndex[s] = [saveIndex]  ###if the key s  doesn't exists for another scenario 
                        saveIndex = -1
            
            ###Add constraint for the loss from a sub-problem solution
            myMasterModel.addConstr(bta[k, s] >= LossExpr, name = "5g"+str(k)+str(s))
            myMasterModel.update() 
            ##Add the excess variable constraint for the sub-problem solution
            myMasterModel.addConstr(v[k,s] >= (LossExpr - eta[k]), name = "5f"+str(k)+str(s))
            myMasterModel.update() 
            
            for j in attNodes:
                N[j].attOA[:] = []      #Empty the attack related list of outgoing arcs
                N[j].child = 0          #Reset the attribute child to 0   
            
            
            
            ##Build the list of Attacks
            #if attDamage > 0:
                #Attacks.append(attack(attDamage))
            
        subModel.params.outputFlag = 1
        ##Add the excess variable v for the current scenario
        #v[k,s] = myMasterModel.addVar(lb=0,name="v"+str(k)+str(s))
        #myMasterModel.update() 
        ##Append the Loss expressions from each scenario to the list  
        #LossExprLst.append(LossExpr)
        ##Write the constraint 5(f)
        
        #myMasterModel.addConstr(v[k,s] >= (LossExpr - eta[k]), name = "5f"+str(k)+str(s))
        #myMasterModel.update() 
        
        ##Build the list of Attacks
        #if attDamage > 0:
        #    Attacks.append(attack(attDamage))
                       
    ##Add the 5(b) constraint for the current iteration
    myMasterModel.addConstr(theta >= quicksum(Scenario_Prob[s] * bta[k, s] for s in range(nScenario)) + Lamda *(eta[k]+(1/(1-Alpha))* quicksum(Scenario_Prob[s]* v[k, s] for s in range(nScenario))))                        
    myMasterModel.update()
    myMasterModel.write("masterModel.lp")
    
    ##########################################
    ##Calculate the risk averse subproblem obj value and update the Upper Bound###############
    #print("List of subproblem solution:", SubSolutionLst)
    ##Sort the list of Subproblem obj values in increasing order
    Sorted_SubSolutionLst = sorted(SubSolutionLst)
    #print ("Sorted subproblem solution:", Sorted_SubSolutionLst)
    ############################################################################
    ##Calculate the nth percentile of the Subproblem obj values
    Losses = np.array(Sorted_SubSolutionLst)
    valueAtRisk = np.percentile(Losses, Value_at_risk)  ##Returns n th percentile
    
    #print("valueAtRisk:", valueAtRisk)
    ##########################################################################
    ##Calculate the excess loss amount
    Excess_Loss = [] 
    for s in range(nScenario):
        Subtract_loss = SubSolutionLst[s]-valueAtRisk
        if Subtract_loss > 0:
            Excess_Loss.append(Subtract_loss)
        else:
            Excess_Loss.append(0)
    ###Calculate the risk averse subproblem obj value
    RiskSubObj = 0
    product1 = 0
    product2 = 0
    for s in range(nScenario):
        product1 = product1+ (Scenario_Prob[s] * SubSolutionLst[s])
        product2 = product2 + (Scenario_Prob[s] * Excess_Loss[s])

    RiskSubObj = product1 + (Lamda*(valueAtRisk + 1/(1-Alpha)*product2)) 
    
    if len(IterAttacks) == k:
        IterAttacks.append(iterAttack(k))
        
    #print("+++++++++++++++++++IterAttacks:",k," -- ", IterAttacks[k], IterAttacks[k].scenarioPathsIndex)
    
    #print("Current subproblem obj value:", RiskSubObj)
    ###############################################################################
    ##Append the SubSolutionLst to the scenarioLoss list of the iterAttack
    IterAttacks[k].scenarioLoss.extend(SubSolutionLst)
    IterAttacks[k].scenarioLossTemp.extend(SubSolutionLst)
    IterAttacks[k].scenarioLossTemp1.extend(SubSolutionLst)
    ##Insert the attackDamage value to each iterattack
    if RiskSubObj > 0:
        IterAttacks[k].attackDamage = RiskSubObj
        IterAttacks[k].attackDamageTemp = RiskSubObj
        IterAttacks[k].attackDamageTemp1 = RiskSubObj  
    

    #Select a set of arcs to interdict using heuristics #####################
    #########################################################################
    #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^bestArcs: ", bestArcs)
    if Hf==1:
            #print("cumArcs before arcSelect",len(cumArcs))
        arcSelect(bestArcs)
    
    
    #Find the arcs in the interdiction plan (find out which arcs are interdicted. This can be done by just getting 
        # the Xval dictionary, that tells what arcs have interdiction variable of 1.
    #####################################################################################################
    defNodes = []
    defArcs = []
    for j in range(len(A)):
            if X[j] >= 0.99:
                defArcs.append(A[j].aIndex)
                if N[A[j].tail].nStatus == 2:
                    defNodes.append(A[j].tail)
                defNodes.append(A[j].head) 
                
    defNodes = set(defNodes)        #Remove the duplicate nodes
        #print("set",defNodes)
        #print(defArcs)
    defArcs = set(defArcs)      #Remove the duplicate arcs
        #print("set",defArcs)
    print("-------------------------------------------------------------------")
    if defArcs in allInterdicts:
        
        
        print("Old defense repeated")# if old defence repeat 2 times, algorithm will converge?
    else:
        print("New defense")
    
    print("------------------------------------------------------------------------")
    
    allInterdicts.append(defArcs)   
        
        
    ###Update the Upper Bound
    if RiskSubObj < UB:
        UB = RiskSubObj
        Xijopt= X   #take the X as the best interdiction plan upto this
    print("Upper Bound:", UB)
    
    print("End of iteration", k)
    print("-------------------------------------------------------------")
    
    ###Update the iteration counter
    k = k+1
    iterCount = iterCount+1
    
    ##Check the optimality condition####
    ###############################################
    if  (UB-LB)/UB <= tolerance:
        print("UB:", UB)
        print("LB:", LB)
        print("Optimal interdiction plan:", Xijopt)
        break
    elif k >= 50:
        print("UB:", UB)
        print("LB:", LB)
        Gap = (UB-LB)/UB 
        print("Optimal interdiction plan:", Xijopt)
        print("Optimality Gap", Gap)
        break
#Retrieve the total run time of the algorithm
algTime = datetime.datetime.now() - t1
algTime = algTime.total_seconds()
print("Total algorithm run time:",algTime)   
print("Master problem time:",masTime)
print("Sub problem time:",subTime)
print("Growth of master problem time:",masTimeAll)
print("Total master problem variables:",masVarAll[-1])
print("Total master problem constraints:",masConstAll[-1])
print("Growth of master problem variables:",masVarAll)
print("Growth of master problem constraints:",masConstAll)
print("Number of nodes:",len(N))
print("Number of arcs:",len(A))
print("Master problem optimal solution and heuristic solution:")

#Write the output file
ofile = open("Output.txt", "a")
ofile.write("\nInput Filename: %s"% filename)
ofile.write("\nNumber of nodes: %s"% len(N))
ofile.write("\nNumber of arcs: %s"% len(A))
ofile.write("\nRisk coefficient: %s"% Lamda)
ofile.write("\nConfidence level: %s"% Alpha)
ofile.write("\nOptimal Interdiction plan: %s"% Xijopt)
ofile.write("\nOptimal Expected risk: %s"% UB)
ofile.write("\nLower Bound: %s"% LB)
ofile.write("\nUpper Bound: %s"% UB)
ofile.write("\nDefender's budget: %s"% Bd)
ofile.write("\nGap: %s"% Gap)
ofile.write("\nTotal algorithm run time: %s" %algTime)
ofile.write("\nMaster problem time:: %s" %masTime)
ofile.write("\nSub problem time: %s" %subTime)
ofile.write("\nGrowth of master problem variables: %s" %masVarAll)
ofile.write("\nGrowth of master problem constraints: %s" %masConstAll)
ofile.write("\n\n----------------------------------------------------------------------------------\n\n")
ofile.write("\nNumber of scenario: %s"% nScenario)
ofile.write("\nAttacker's budget: %s"% Ba)
ofile.write("\nMean prob:%s" %MeanProb)
ofile.write("\nScenario probabilities:%s" %Scenario_Prob)
ofile.write("\n\n----------------------------------------------------------------------------------\n\n")
ofile.write("\n\n----------------------------------------------------------------------------------\n\n")
ofile.close()


    
    
                                
                        
                
                
                
                    
                    
            
            
        
        
        
        
     
    
    
    
        
    












    
    


           

