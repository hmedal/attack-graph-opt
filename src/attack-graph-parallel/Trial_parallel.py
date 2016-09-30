'''
Created on Sep 19, 2016

@author: tb2038
'''

import random
import datetime
from gurobipy import *
import copy
import math
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
numprocs = comm.size
rank =comm.Get_rank()

nScenario= 100.0    #No of Scenarios
tolerance= 0.05 #gap between the upper and lower bound
SD = 0.05       #standard deviation
Bd= 150.0           #Defender's budget
Ba= []  #Attacker's budget
Gap = 80.0
###Define the risk parameters
Lamda = 0.1
Alpha = 0.9
Value_at_risk = 90.0 

##Control parameters for enhancements of CCG algorithm
Hi = 1      #Heuristic for initial master problem solution
Ms = 1      #Add multiple sub problem solution to the master problem
TR = 1      #Add trust region cut to the master problem
Hf = 1      #Apply the final heuristic

masterModel = Model("master")
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
def calcMasterproblem (masterModel, theta, x, m):
    global Hi
    if m == 0:
        ###Add theta variable
        theta = masterModel.addVar(lb=0, ub=GRB.INFINITY, obj=1, name='theta')
        ##Define the interdiction variable
        x = {}
        for a in range(len(A)):
            x[a] = masterModel.addVar(vtype=GRB.BINARY, name='x'+str(A[a].tail)+str(A[a].head))
            ##Update the model
        masterModel.update()
        masterModel.addConstr(quicksum(A[a].securityCost* x[a] for a in range(len(A))) <= Bd)
        m = m + 1

    masterModel.update()
    masterModel.modelSense= GRB.MINIMIZE
    #masterModel.write("masterModel.lp")
    masterModel.setParam("OutputFlag", 0)
    masterModel.optimize() 
    
    ########declare the dictionary of the interdiction values
    xval={}
    xval= masterModel.getAttr('x', x)
    
    etaVal = {}
    etaVal=masterModel.getAttr('x', eta)
    
    vVal = {}
    vVal = masterModel.getAttr('x', v)
    
    uVal = {}
    uVal = masterModel.getAttr('x', u)
    
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
    
    
    return xval, masterModel.objVal,theta, x, m, etaVal, vVal, uVal 
    
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
filename="Tnet100-3t5.txt"
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
    Ba.append(random.uniform(150.0, 200.0))
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
#Define the function to Select a set of arcs to interdict using heuristics ##################### 

def arcSelect(BbestArcs):
    interdictPaths = []
    tCost = 0       #Keep track of the total interdiction cost
    Allarcs = copy.copy(BbestArcs)   
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
BbestArcs = list() #List of all bestarcs from all the processors
saveIndex = -1


k = 0 # iteration counter

##############Start of CCG Algorithm##############################################################
##################################################################################################

# create the master model and write it to masterModel.lp

while UB-LB >= tolerance * UB:
    
    ####Beginning of Solving the master problem loop###############################################
    myMasterModel = read("masterModel.lp")
    X, LB_temp , theta, x,m, etaVal, vVal, uVal = calcMasterproblem(masterModel, theta, x, m)
    ###Calculating the time to solve the master problem and the number of variables and constraints in the masterModel in this iterations
    if iterCount<TRiter:
        LB = 0.0
    else:
        LB = LB_temp  
    print("Lower Bound:", LB)
    
    #Remove the master problem solution stabilizing constraint
    if k !=0:    
        if iterCount<TRiter:
            masterModel.remove(masStableConst)
            masterModel.update()
    SubSolutionLst = []  ##List of the subproblem objective function values from different scenarios
    
    A = math.ceil((nScenario/numprocs))
    Start = rank*A
    End = Start + A
    #numSolLst = []
    #attNIInEachSol = []
    #ListOf_attNI = []
    total_attNodes = {}  #All attack nodes in this iteration.
    total_attNI = {}  #All attack nodes in this iteration.
    total_attOA = {}  # All the outgoing arcs from each node used in the attack of a solution in each scenario
    nSolutionList=[]
    for s in range(Start, End):
        subModel, subobj, wVal, zVal, solCount, w = calcForEachScenario(s, X)
        SubSolutionLst.append(subobj)
        numSols = round(fracSols*solCount)
        #numSolLst.append(numSols)         
        if numSols == 0:
            numSols = 1
        
        nSolutionList.append(numSols)
        subModel.params.outputFlag = 0
        
        for sol in range(int(0), int(numSols)):
            
            attNodes = []   #Nodes used in this attack
            attArcs = []    #Arcs used in this attack
            attNI = []      #Initially vulnerable nodes used in this attack
            attNG = []      #Goal nodes used in this attack
            newArcs = set()     #New arcs used in this attack
            
            subModel.params.SolutionNumber = sol
            
            wVal = subModel.getAttr('Xn', w)
            for a in range(len(A)):
                if wVal[a] >= 0.99:
                    attArcs.append(A[a].aIndex)
                    if N[A[a].tail].nStatus == 2:
                        attNodes.append(A[a].tail)
                        attNI.append(A[a].tail)
                    attNodes.append(A[a].head)
                    #N[A[a].tail].attOA.append(A[a].aIndex)
                    
                    if (s, sol, N[A[a].tail]) in total_attOA:
                        total_attOA[s, sol, N[A[a].tail]].append(A[a].aIndex)
                    else:
                        total_attOA[s, sol, N[A[a].tail]] = []
                        total_attOA[s, sol, N[A[a].tail]].append(A[a].aIndex)
                    if N[A[a].head].nStatus == 1:
                        attNG.append(A[a].head)
                        
            # we use set to remove the duplicate items             
            attNodes = set(attNodes)        #Remove the duplicate nodes
            attNI = set(attNI)
            attNG = set(attNG)
            attArcs = set(attArcs)   #Remove the duplicate arcs
            
            total_attNodes[s, sol] =  attNodes
            total_attNI[s, sol] = attNI     
            
            ##Create the list to send to the rank ==0
            #attNIInEachSol.append(len(attNI))
            #ListOf_attNI = ListOf_attNI.extend(attNI) 
            
            if sol == 0:
                for a in attArcs:
                    bestArcs.append(a)  
                bestArcs = set(bestArcs)
                bestArcs = list(bestArcs)
            
        subModel.params.outputFlag = 1
        
    if rank != 0:
        comm.send(total_attOA, dest=0, tag="total_attOA")
        comm.send(total_attNodes, dest=0, tag="total_attNodes")
        comm.send(total_attNI, dest=0, tag="total_attNI")
        comm.send(bestArcs, dest=0, tag="bestArcs")
        comm.send(SubSolutionLst, dest=0, tag="SubSolutionLst")
        comm.send(nSolutionList, dest = 0, tag = "nSolutionList") 
    comm.Barrier()
    
    if rank == 0:
        Btotal_attOA = list()
        Btotal_attNodes = list()
        Btotal_attNI = list()
        BSubSolutionLst = list()
        BnSolutionList = list()
        
        #Receiving data from rank = 0 itself
        Btotal_attOA.append(total_attOA)  
        Btotal_attNodes.append(total_attNodes)  
        Btotal_attNI.append(total_attNI)
        BbestArcs.extend(bestArcs)
        BSubSolutionLst.extend(SubSolutionLst)
        BnSolutionList.extend(nSolutionList)
        #Receiving data from other processors
        for i in range(1,numprocs):
            total_attOA = comm.recv(source=i, tag="total_attOA")
            total_attNodes = comm.recv(source=i, tag="total_attNodes")
            total_attNI = comm.recv(source=i, tag="total_attNI")
            bestArcs = comm.recv(source=i, tag="bestArcs")
            SubSolutionLst = comm.recv(source=i, tag="SubSolutionLst")
            nSolutionList = comm.recv(source=i, tag="nSolutionList")
            
            Btotal_attOA.append(total_attOA)
            Btotal_attNodes.append(total_attNodes)
            Btotal_attNI.append(total_attNI)
            BbestArcs.extend(bestArcs)
            BSubSolutionLst.extend(SubSolutionLst)
            BnSolutionList.extend(nSolutionList)
        
        BbestArcs = set(BbestArcs)
        BbestArcs = list(BbestArcs)
        
        #Add the eta variable for the current iteration
        eta[k] = masterModel.addVar(lb = 0, ub = GRB.INFINITY, name = "eta"+str(k)) 
        masterModel.update()
        NP = 0  #Define the index of processor
        for s in range(int(nScenario)):
            ##Add the excess variable v for the current scenario
            v[k,s] = masterModel.addVar(lb=0,name="v"+str(k)+str(s))    
    
            ##Add the beta variable for each scenario#############
            bta[k, s] = masterModel.addVar(lb =0, name = "bta"+str(k)+str(s))
            masterModel.update() 
            
            for sol in range(BnSolutionList[s]):
                attPath = []    #List of attack paths used in this attack
                LossExpr = 0    #Expected loss from an attack plan (solution of each scenario)
                dict = Btotal_attNI[NP]
                if (s, sol) in dict:
                    for j in dict[s, sol]:         #Explore the trees of all the initially vulnerable nodes
                        dict_attOA = Btotal_attOA[NP]
                        if (s, sol, j) in dict_attOA:
                            attOA_j = dict_attOA[s, sol, j]
                        tempNode = j
                        treeNotExplored = True
                        while treeNotExplored:      #Explore the tree of an initially vulnerable node to find all the paths
                            dict1 = Btotal_attOA[NP]
                            if (s, sol, tempNode) in dict1:
                                attOA = dict1[s, sol, tempNode]
                            if N[tempNode].child < len(attOA):  # if the index of child of that node is less than the number of outgoing arcs from that node used in this attack
                                attPath.append(attOA[N[tempNode].child])
                                N[tempNode].child = N[tempNode].child+1    #Go to the next child which is another branch
                                if N[A[attPath[-1]].head].nStatus == 1:    #Check if a goal node (leaf) is found 
                                    attPathTemp = copy.copy(attPath)
                                    PathProb = 1.0  
                                    for a in attPathTemp: ##takes values from the list attPathTemp
                                        PathProb = PathProb * MeanProb[a]
                                    if attPathTemp in AllAttPaths:
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
                                        u[len(AllAttPaths)-1] = masterModel.addVar(lb=0,ub=1,name="u"+str(len(AllAttPaths)-1))
                                        masterModel.update()
                                        
                                        LossExpr = LossExpr + N[A[attPath[-1]].head].bLoss*PathProb*(1-u[len(AllAttPaths)-1])
                                        #Add constraint (5c) to the masterModel
                                        masterModel.addConstr(u[len(AllAttPaths)-1] <= quicksum(x[l] for l in attPathTemp),"5c"+str(len(AllAttPaths)-1))
                                
                                tempNode = A[attPath[-1]].head
                            else:
                                tempNode = A[attPath[-1]].tail
                                attPath.pop()
                                        
                            if len(attPath)==0 and N[j].child>=len(attOA_j):
                                treeNotExplored = False
                                
                            if sol == 0 and saveIndex>-1:
                                bestPathsIndices.append(saveIndex)
                                bestPathsIndices = set(bestPathsIndices)
                                bestPathsIndices = list(bestPathsIndices)
                                if len(IterAttacks) == k:#New entry for this iteration
                                    IterAttacks.append(iterAttack(k))
                                    IterAttacks[k].scenarioPathsIndex[s] = [saveIndex]
                                else:
                                    if s in IterAttacks[k].scenarioPathsIndex:
                                        IterAttacks[k].scenarioPathsIndex[s].append(saveIndex)
                                    else:
                                        IterAttacks[k].scenarioPathsIndex[s] = [saveIndex]
                                    saveIndex = -1
                                    
                    masterModel.addConstr(bta[k, s] >= LossExpr, name = "5g"+str(k)+str(s)) 
                    masterModel.addConstr(v[k,s] >= (LossExpr - eta[k]), name = "5f"+str(k)+str(s))
                    masterModel.update() 
                    
                    dict2 = Btotal_attNodes[NP]
                    if (s, sol) in dict2: 
                        for j in dict2[s, sol]:
                            #N[j].attOA[:] = []
                            N[j].child = 0                    
                                            
                else:
                    NP = NP +1
                    sol = -1  
                if NP == numprocs:
                    break
                                              
            
            
            if NP == numprocs:
                break
                                                               
        masterModel.addConstr(theta >= quicksum(Scenario_Prob[s] * bta[k, s] for s in range(int(nScenario))) + Lamda *(eta[k]+(1/(1-Alpha))* quicksum(Scenario_Prob[s]* v[k, s] for s in range(int(nScenario)))))                        
        masterModel.update()
        masterModel.write("masterModel.lp") 
        Sorted_SubSolutionLst = sorted(BSubSolutionLst)
        
        Losses = np.array(Sorted_SubSolutionLst)
        valueAtRisk = np.percentile(Losses, Value_at_risk)
        
        Excess_Loss = [] 
        for s in range(int(nScenario)):
            Subtract_loss = BSubSolutionLst[s]-valueAtRisk
            if Subtract_loss > 0:
                Excess_Loss.append(Subtract_loss)
            else:
                Excess_Loss.append(0)
                                
        RiskSubObj = 0
        product1 = 0
        product2 = 0 
        for s in range(int(nScenario)):
            product1 = product1+ (Scenario_Prob[s] * BSubSolutionLst[s])
            product2 = product2 + (Scenario_Prob[s] * Excess_Loss[s])                                       
        RiskSubObj = product1 + (Lamda*(valueAtRisk + 1/(1-Alpha)*product2)) 
        
        if len(IterAttacks) == k:
            IterAttacks.append(iterAttack(k))
        
        
        IterAttacks[k].scenarioLoss.extend(BSubSolutionLst)
        IterAttacks[k].scenarioLossTemp.extend(BSubSolutionLst)
        IterAttacks[k].scenarioLossTemp1.extend(BSubSolutionLst)
                                            
        if RiskSubObj > 0:
            IterAttacks[k].attackDamage = RiskSubObj
            IterAttacks[k].attackDamageTemp = RiskSubObj
            IterAttacks[k].attackDamageTemp1 = RiskSubObj 
        
        if Hf==1:
            #print("cumArcs before arcSelect",len(cumArcs))
            arcSelect(bestArcs)
            
        
        if iterCount<TRiter:
            masStableExpr = 0
            countMasOnes = 0
            for j in range(len(A)):
                if X[j]>0.99:   #need CHANGE AFTER THE DR MEDAL'S WORK
                    masStableExpr = masStableExpr+1-x[j]
                    countMasOnes = countMasOnes+1
                else:
                    masStableExpr = masStableExpr+x[j]
            masStableConst = masterModel.addConstr(masStableExpr <= masMaxJump*2*countMasOnes, "(TR)"+str(iterCount))
            masterModel.update()
        
        defNodes = []
        defArcs = []
        for j in range(len(A)):
            if X[j] >= 0.99:
                defArcs.append(A[j].aIndex)
                if N[A[j].tail].nStatus == 2:
                    defNodes.append(A[j].tail)
                defNodes.append(A[j].head)
        
        defNodes = set(defNodes)
        
        if defArcs in allInterdicts:
            print("Old defense repeated")
        
        else:
            print("New defense")
        
        allInterdicts.append(defArcs)
        
        if RiskSubObj < UB:
            UB = RiskSubObj
            Xijopt= X   #take the X as the best interdiction plan upto this
        print("Upper Bound:", UB)
    
        print("End of iteration", k)
        print("-------------------------------------------------------------")
        k = k+1
        iterCount = iterCount+1 
        
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
    comm.Barrier()

if rank==0:
    print("Number of nodes:",len(N))
    print("Number of arcs:",len(A))
    
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
    ofile.write("\n\n----------------------------------------------------------------------------------\n\n")
    ofile.write("\nNumber of scenario: %s"% nScenario)
    ofile.write("\nAttacker's budget: %s"% Ba)
    ofile.write("\nMean prob:%s" %MeanProb)
    ofile.write("\nScenario probabilities:%s" %Scenario_Prob)
    ofile.write("\n\n----------------------------------------------------------------------------------\n\n")
    ofile.write("\n\n----------------------------------------------------------------------------------\n\n")
    ofile.close()    
    
MPI.Finalize()    
            