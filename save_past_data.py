# -*- coding: utf-8 -*-

import pandas as pd
import urllib
import os, os.path
import time


def main():

    if not os.path.exists("Data/main"):
        os.makedirs("Data/main")

    df = pd.read_csv("Data/tosyo1_list.csv", encoding="SHIFT-JIS")
    for i, record in df.iterrows():
        if i < 432:
            continue
        print i, record[u"コード"], record[u"銘柄名"]
        code = record[u"コード"]
        name = record[u"銘柄名"]
        data = pd.DataFrame()
        for year in ["2017", "2016", "2015", "2014", "2013"]:
            url = "http://k-db.com/stocks/" + code + "/1d/" + year + "?download=csv"
            is_done = False
            while not is_done:
                try:
                    a, b = urllib.urlretrieve(url, "Data/tmp.csv")
                    time.sleep(5)
                    fn = b.dict["content-disposition"].split(";")[1].split("=")[1]
                    is_done = True
                except Exception as e:
                    print "Error: invalid data"
                    print e
            if "1d" not in fn:
                break
            df_tmp = pd.read_csv("Data/tmp.csv", encoding="SHIFT-JIS")
            data = data.append(df_tmp)

        data.to_csv("Data/main/"+code+"_"+name+".csv", index=False, encoding="SHIFT-JIS")


if __name__ == "__main__":
    main()
