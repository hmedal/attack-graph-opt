'''
Created on Sep 30, 2016

@author: hm568
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a filename.')
    parser.add_argument('-m','--masterproblem', help='the master problem file')
    args = parser.parse_args()
