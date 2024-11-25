import sys
from plot_dependencies import basic_line

if len(sys.argv) > 1:
    print(sys.argv[1])
    basic_line(csv_file=sys.argv[1])
else:
    print("Expecting csv file name of manual labels")
    exit()
