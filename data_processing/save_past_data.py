# -*- coding: utf-8 -*-

import pandas as pd
import urllib
import os, os.path
import time

import config as c


year_list =  ["2017", "2016", "2015", "2014", "2013"]


def save_individual():

    if not os.path.exists(os.path.join(c.data_root, "main")):
        os.makedirs(os.path.join(c.data_root, "main"))

    df = pd.read_csv(os.path.join(c.data_root, "tosyo1_list.csv"), encoding="SHIFT-JIS")
    for i, record in df.iterrows():
        print i, record[u"コード"], record[u"銘柄名"]
        code = record[u"コード"]
        name = record[u"銘柄名"]
        data = pd.DataFrame()
        for year in year_list:
            url = "http://k-db.com/stocks/" + code + "/1d/" + year + "?download=csv"
            is_done = False
            while not is_done:
                try:
                    a, b = urllib.urlretrieve(url, "tmp.csv")
                    time.sleep(3)
                    fn = b.dict["content-disposition"].split(";")[1].split("=")[1]
                    is_done = True
                except Exception as e:
                    print "Error: invalid data"
                    print e
            if "1d" not in fn:
                break
            df_tmp = pd.read_csv("tmp.csv", encoding="SHIFT-JIS")
            data = data.append(df_tmp)

        data.to_csv(os.path.join(c.data_root, "main", code+"_"+name+".csv"), index=False, encoding="SHIFT-JIS")


def save_indices():

    if not os.path.exists(os.path.join(c.data_root, "indices")):
        os.makedirs(os.path.join(c.data_root, "indices"))

    df = pd.read_csv(os.path.join(c.data_root, "indices_list.csv"), encoding="SHIFT-JIS")
    for i, record in df.iterrows():
        print i, record[u"指数"], record[u"id"]
        name = record[u"指数"]
        code = record[u"id"]
        data = pd.DataFrame()
        for year in year_list:
            url = "http://k-db.com/indices/" + code + "/1d/" + year + "?download=csv"
            is_done = False
            while not is_done:
                try:
                    a, b = urllib.urlretrieve(url, "tmp.csv")
                    time.sleep(3)
                    fn = b.dict["content-disposition"].split(";")[1].split("=")[1]
                    is_done = True
                except Exception as e:
                    print "Error: invalid data"
                    print e
            if "1d" not in fn:
                break
            df_tmp = pd.read_csv("tmp.csv", encoding="SHIFT-JIS")
            data = data.append(df_tmp)

        data.to_csv(os.path.join(c.data_root, "indices", code+"_"+name+".csv"), index=False, encoding="SHIFT-JIS")

if __name__ == "__main__":
    #save_individual()
    save_indices()
