import sys
import pandas as pd

if __name__ == '__main__':
    feth_filename = sys.argv[1]
    df = pd.read_feather(feth_filename)
    df.to_excel(feth_filename + ".xlsx")