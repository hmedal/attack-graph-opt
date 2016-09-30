import os

serverName = 'hmedal@shadow-login'
codePathOnCluster = '/work/hmedal/code/src/python/'
destFullPath = serverName + ':' + codePathOnCluster
sourcePath = '/Users/hm568/Google\ Drive/Documents/2_msu/research_manager/code/repos/attack-graph-opt'

rsyncStr = 'rsync -av ' + sourcePath + ' ' + destFullPath
os.system(rsyncStr)

serverName = 'hmedal@transfer'
codePathOnCluster = '/dasi/projects/idaho-bailiff/G7/code/src/python'
destFullPath = serverName + ':' + codePathOnCluster
sourcePath = '/Users/hm568/Google\ Drive/Documents/2_msu/research_manager/code/repos/attack-graph-opt'

rsyncStr = 'rsync -av ' + sourcePath + ' ' + destFullPath
os.system(rsyncStr)