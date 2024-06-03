"""
cat outputs_old |grep -E "^Average|^\||^RESULTS"

"""
import re
import sys


def get_results(lines):
    name = lines[0].split(':')[1].strip()
    map25 = lines[1].split('=')[2].replace('(', '').replace('}', '').replace(')', '%').replace(' ', '').replace('%', '')
    map50 = lines[2].split('=')[2].replace('(', '').replace('}', '').replace(')', '%').replace(' ', '').replace('%', '')
    map100 = lines[3].split('=')[2].replace('(', '').replace('}', '').replace(')', '%').replace(' ', '').replace('%', '')
    map = lines[4].split(':')[1].replace('(', '').replace('}', '').replace(')', '%').replace(' ', '').replace('%', '')
    return name, map25, map50, map100, map


f = sys.argv[1]
with open(f) as fi:
    text = [t.strip('\n') for t in fi.readlines()]

for i in range(3):
    lines = text[5*i:5*i+5]
    print(get_results(lines))

