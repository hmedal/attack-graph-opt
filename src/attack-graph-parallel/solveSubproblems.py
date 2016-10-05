'''
Created on Sep 30, 2016

@author: hm568
'''
import random
import datetime
from gurobipy import *
import copy
import math
import numpy as np
import attackGraph
from mpi4py import MPI
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a filename.')
    parser.add_argument('-m','--masterproblem', help='the master problem file')
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    numprocs = comm.size
    rank =comm.Get_rank()
    ####Beginning of Solving the master problem loop###############################################
    myMasterModel = read("masterModel.lp")
    theta = myMasterModel.getVars()[0]
    x = myMasterModel.getVars()[1:numArcs] # change the indices
    beta = myMasterModel.getVars()[1:numArcs] # change the indices
    B = myMasterModel.getVars()[1:numArcs] # change the indices
    X, LB_temp , theta, x,m, etaVal, vVal, uVal = calcMasterproblem(myMasterModel, theta, x, m)
    ###Calculating the time to solve the master problem and the number of variables and constraints in the masterModel in this iterations
    if iterCount<TRiter:
        LB = 0.0
    else:
        LB = LB_temp  
    print("Lower Bound:", LB)
    
    '''
    #Remove the master problem solution stabilizing constraint
    if k !=0:    
        if iterCount<TRiter:
            myMasterModel.remove(masStableConst)
            myMasterModel.update()
    '''
    
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
        eta[k] = myMasterModel.addVar(lb = 0, ub = GRB.INFINITY, name = "eta"+str(k)) 
        myMasterModel.update()
        NP = 0  #Define the index of processor
        for s in range(int(nScenario)):
            ##Add the excess variable v for the current scenario
            v[k,s] = myMasterModel.addVar(lb=0,name="v"+str(k)+str(s))    
    
            ##Add the beta variable for each scenario#############
            bta[k, s] = myMasterModel.addVar(lb =0, name = "bta"+str(k)+str(s))
            myMasterModel.update() 
            
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
                                        u[len(AllAttPaths)-1] = myMasterModel.addVar(lb=0,ub=1,name="u"+str(len(AllAttPaths)-1))
                                        myMasterModel.update()
                                        
                                        LossExpr = LossExpr + N[A[attPath[-1]].head].bLoss*PathProb*(1-u[len(AllAttPaths)-1])
                                        #Add constraint (5c) to the myMasterModel
                                        myMasterModel.addConstr(u[len(AllAttPaths)-1] <= quicksum(x[l] for l in attPathTemp),"5c"+str(len(AllAttPaths)-1))
                                
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
                                    
                    myMasterModel.addConstr(bta[k, s] >= LossExpr, name = "5g"+str(k)+str(s)) 
                    myMasterModel.addConstr(v[k,s] >= (LossExpr - eta[k]), name = "5f"+str(k)+str(s))
                    myMasterModel.update() 
                    
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
                                                               
        myMasterModel.addConstr(theta >= quicksum(Scenario_Prob[s] * bta[k, s] for s in range(int(nScenario))) + Lamda *(eta[k]+(1/(1-Alpha))* quicksum(Scenario_Prob[s]* v[k, s] for s in range(int(nScenario)))))                        
        myMasterModel.update()
        myMasterModel.write("masterModel.lp") 
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
            
        '''
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
        '''
            
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
    
