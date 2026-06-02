import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/Accounts/localdawsonj2/Desktop/HumanFollowingTurtlebot4/install/human_detector'
