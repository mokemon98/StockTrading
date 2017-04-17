# -*- coding: utf-8 -*-

import pandas as pd
import urllib
import os, os.path
import shutil

import config as c


def update_individual():

    url = "http://k-db.com/stocks/?download=csv"
    #url = "http://k-db.com/stocks/2017-04-07?download=csv"
    a, b = urllib.urlretrieve(url, "tmp.csv")
    name = b.dict["content-disposition"].split(";")[1].split("=")[1]
    if not name.startswith("stocks"):
        print "Error: failed to get a daily file from web"
        print "name:", name
        exit()

    if name in os.listdir(os.path.join(c.data_root, "backup")):
        print "Waning: downloaded data has already been in backup directory"

    shutil.copyfile("tmp.csv", os.path.join(c.data_root, "backup", name))

    day_str = name.split("_")[1].split(".")[0]

    df = pd.read_csv("tmp.csv", encoding="SHIFT-JIS")
    tosyo1 = df[df[u"市場"]==u"東証1部"]
    for i, r in tosyo1.iterrows():
        #print i, r[u"コード"], r[u"銘柄名"]
        code = r[u"コード"]
        name = r[u"銘柄名"]
        df1 = pd.DataFrame([[day_str, r[3], r[4], r[5], r[6], r[7], r[8]]],
            columns=[u"日付", u"始値", u"高値", u"安値", u"終値", u"出来高", u"売買代金"])
        fn = os.path.join(c.data_root, "main", code+"_"+name+".csv")
        try:
            df2 = pd.read_csv(fn, encoding="SHIFT-JIS")
        except IOError:
            print i, code, name, "Error: File doesn't exist."
            continue
        except Exception as e:
            print type(e), e.message
            exit()

        if len(df2[df2[u"日付"]==day_str]) == 0:
            df3 = df1.append(df2)
            df3 = df3.sort_values(by=u"日付", ascending=False)
            df3.to_csv(fn, index=False, encoding="SHIFT-JIS")


def update_indices():

    url = "http://k-db.com/indices/?download=csv"
    a, b = urllib.urlretrieve(url, "tmp.csv")
    name = b.dict["content-disposition"].split(";")[1].split("=")[1]
    if not name.startswith("indices"):
        print "Error: failed to get a daily file from web"
        print "name:", name
        exit()

    if name in os.listdir(os.path.join(c.data_root, "backup")):
        print "Waning: downloaded data has already been in backup directory"

    shutil.copyfile("tmp.csv", os.path.join(c.data_root, "backup", name))

    day_str = name.split("_")[1].split(".")[0]

    df = pd.read_csv("tmp.csv", encoding="SHIFT-JIS")
    df_code = pd.read_csv(os.path.join(c.data_root, "indices_list.csv"), encoding="SHIFT-JIS")
    for i, r in df.iterrows():
        name = r[u"指数"]
        temp = df_code[df_code[u"指数"]==name]["id"]
        if len(temp) == 0:
            continue
        else:
            code = temp.values[0]
        print name, code
        df1 = pd.DataFrame([[day_str, r[1], r[2], r[3], r[4]]],
            columns=[u"日付", u"始値", u"高値", u"安値", u"終値"])
        fn = os.path.join(c.data_root, "indices", code+"_"+name+".csv")
        try:
            df2 = pd.read_csv(fn, encoding="SHIFT-JIS")
        except IOError:
            print i, code, name, "Error: File doesn't exist."
            continue
        except Exception as e:
            print type(e), e.message
            exit()

        if len(df2[df2[u"日付"]==day_str]) == 0:
            df3 = df1.append(df2)
            df3 = df3.sort_values(by=u"日付", ascending=False)
            df3.to_csv(fn, index=False, encoding="SHIFT-JIS")


if __name__ == "__main__":
    update_individual()
    update_indices()
