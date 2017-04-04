# -*- coding: utf-8 -*-

import pandas as pd
import urllib
import os, os.path
import shutil


def main():

    url = "http://k-db.com/stocks/?download=csv"
    a, b = urllib.urlretrieve(url, "Data/tmp.csv")
    name = b.dict["content-disposition"].split(";")[1].split("=")[1]
    if not name.startswith("stocks"):
        print "Error: failed to get a daily file from web"
        exit()

    files = os.listdir("Data/backup")

    if name in files:
        print "Waning: downloaded data has already been in backup directory"
        #exit()

    shutil.copyfile("Data/tmp.csv", "Data/backup/"+name)

    day_str = name.split("_")[1].split(".")[0]

    df = pd.read_csv("Data/tmp.csv", encoding="SHIFT-JIS")
    tosyo1 = df[df[u"市場"]==u"東証1部"]
    for i, r in tosyo1.iterrows():
        print i, r[u"コード"], r[u"銘柄名"]
        code = r[u"コード"]
        name = r[u"銘柄名"]
        df1 = pd.DataFrame([[day_str, r[3], r[4], r[5], r[6], r[7], r[8]]],
            columns=[u"日付", u"始値", u"高値", u"安値", u"終値", u"出来高", u"売買代金"])
        fn = "Data/main/"+code+"_"+name+".csv"
        df2 = pd.read_csv(fn, encoding="SHIFT-JIS")
        if len(df2[df2[u"日付"]==day_str]) == 0:
            df3 = df1.append(df2)
            df3.to_csv(fn, index=False, encoding="SHIFT-JIS")


if __name__ == "__main__":
    main()
