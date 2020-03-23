import sys
import os

macro_f1 = 0
micro_f1 = 0
a = 0
b = 0

input_file = sys.argv[1]
fi = open(input_file, 'r')
for line in fi:
    if line[0:9] == 'macro-f1:':
        macro_f1 = macro_f1 + float(line.split(':')[1])
        a = a + 1
    if line[0:9] == 'micro-f1:':
        micro_f1 = micro_f1 + float(line.split(':')[1])
        b = b + 1
fi.close()

macro_f1 = macro_f1 / a
micro_f1 = micro_f1 / b

print("Macro-F1: " + str(macro_f1))
print("Micro-F1: " + str(micro_f1))
