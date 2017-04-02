# -*- coding: utf-8 -*-

import pandas as pd


def main():

    df = pd.read_csv("Data\stocks_2017-03-31.csv", encoding="SHIFT-JIS")
    tosyo1 = df[df[u"市場"]==u"東証1部"]
    tosyo1_2 = tosyo1[[u"コード", u"銘柄名", u"市場"]]
    tosyo1_2.to_csv("Data/tosyo1_list.csv", index=False, encoding="SHIFT-JIS")
    #from IPython import embed; embed()


if __name__ == "__main__":
    main()
