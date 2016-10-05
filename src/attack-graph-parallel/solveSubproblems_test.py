import unittest
import attackGraph

class TestSolvesubproblems(unittest.TestCase):
    
    '''
    This class is for testing the different scheduleable set classes.
    '''
    
    '''
    THIS IS FOR THE ADDITIVE MODEL
    '''
    
    def testSolveSubproblems(self):
        os.system('python solveSubproblems.py master_' + str(iter) + '.sol') # solve sub problem, given solution to master problem