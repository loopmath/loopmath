import sys
print(sum(float(x) for x in sys.argv[1:] if float(x) > 0))
