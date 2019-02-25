'''
file_list = ["train-queries", "train-purchases", "train-item-views", "train-clicks", "products", "product-categories"]
            #[923,127,          18,025,             1,235,380,          1,127,764,      184,047,    184,047]

for file in file_list:
    num = 0
    with open(file+".csv", "r") as fin:
        fin.readline()
        for line in fin:
            num += 1
    print "num of lines of "+ file + ":", num
'''

query_click = {}
linenum = 0
with open("train-clicks.csv", "r") as fin:
    fin.readline()
    for line in fin:
        linenum += 1
        tmp = line.strip().split(";")
        if len(tmp[2].strip().split()) != 1:
            print line
        if int(tmp[0]) in query_click:
            query_click[int(tmp[0])].append(int(tmp[2]))
        else:
            query_click[int(tmp[0])] = [int(tmp[2])]

print len(query_click) # 633,732
m = 0
for q in query_click:
    m += len(query_click[q])
print "query_click:", linenum, m, float(m) / len(query_click)    # 3,728,507     5.88

session_purchase = {}
linenum = 0
with open("train-purchases.csv", "r") as fin:
    fin.readline()
    for line in fin:
        linenum += 1
        tmp = line.strip().split(";")    
        if len(tmp[-1].split()) > 1:
            print tmp
            exit()
        if int(tmp[0]) in session_purchase:
            session_purchase[int(tmp[0])].append(tmp[-1])
        else:
            session_purchase[int(tmp[0])] = [tmp[-1]]
print len(session_purchase)  # 12,630
m = 0
for q in session_purchase:
    m += len(session_purchase[q])
print "train-purchases:", linenum, m, float(m) / len(session_purchase)    # 18,025  1.43

session = {}
with open("train-sci.csv", "w") as fout:
    with open("train-queries.csv", "r") as fin:
        fin.readline()
        linenum = 0
        for line in fin:
            linenum += 1
            tmp = line.strip().split(";")
            #if tmp[-1] == "FALSE":
            if int(tmp[1]) in session:
                session[int(tmp[1])].append({"queryid":int(tmp[0]), "itemlist":tmp[8]})
            else:
                session[int(tmp[1])] = [{"queryid":int(tmp[0]), "itemlist":tmp[8]}]
        print len(session)  # 368,782 train+test: 573,935
        m = 0
        for q in session:
            m += len(session[q])
        print linenum, m, float(m) / len(session)   # 923,127 1.73
    print >> fout, "session_id;query_id;clickitem;itemlist;purchase"
    slist = session.keys()
    session_num, query_result, query_no_result = 0, 0, 0
    for s in sorted(slist):
        #if len(session[s]) > 1:
        session_num += 1
        for k, item in enumerate(session[s]):
            if item["queryid"] in query_click:
                print >> fout, str(s)+";"+str(item["queryid"])+";"+','.join(map(str, query_click[item["queryid"]]))+";"+item["itemlist"]+";",
                #if s == 150:
                    #print str(query_click[item["queryid"]][0]), session_purchase[s]
                if s in session_purchase and len(query_click[item["queryid"]]) == 1:                    
                    if str(query_click[item["queryid"]][0]) in session_purchase[s]:
                        print >> fout, "1"
                    else:
                        print >> fout, "0"
                else:
                    print >> fout, '0'
                query_result += 1
            else:
                query_no_result += 1
        #if tmp != []:
            #print s



print session_num, query_result, query_no_result   #119,415    after removing query without click: 119,366
#sci means "session, click, item", 1c means just one click at each step
session = {}
with open("train-sci-1c-1p.csv", "w") as fout:
    with open("train-sci.csv", "r") as fin:
        fin.readline()
        for line in fin:
            tmp = line.strip().split(";")
            if len(tmp[2].strip().split(",")) == 1 and len(tmp[-1].strip().split(",")) == 1:
                print >> fout, line.strip()
                if tmp[0] not in session:
                    session[tmp[0]] = 1
print len(session)  #103,054

session = {}
with open("train-sci-1c.csv", "w") as fout:
    with open("train-sci.csv", "r") as fin:
        fin.readline()
        for line in fin:
            tmp = line.strip().split(";")
            if len(tmp[2].strip().split(",")) == 1:
                print >> fout, line.strip()
                if tmp[0] not in session:
                    session[tmp[0]] = 1
print len(session)      #105,089

