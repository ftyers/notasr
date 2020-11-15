import sys 
import random
v = open('/tmp/vocab').read().strip().split(' ')
for i in range(0, int(sys.argv[1])):
    print(' '.join([v[random.randint(0,4)] for i in range(0, 5)]))
