path = "../ZFB-output"

for i in range(1, 6):
    seq = str(i)
    print('file {}'.format(i))
    fnr = path + "/ZFB-" + seq + "-Selected.csv"
    fnw = path + "/ZFB-" + seq + "-Cleaned.csv"

    fw = open(fnw, 'w')
    klines = 0
    with open(fnr, 'r') as csvfile:
        for line in csvfile:
            klines += 1

            idx = 0
            for fld in line.split(","):
                idx += 1
                fld = fld.strip()
                if "/" in fld:
                    if idx == 2:
                        fw.write(fld + ",")
                    if idx > 4:
                        fw.write(",")
                else:
                    fw.write(fld + ",")

            fw.write("\n")
    fw.close()
print('Execution finished')
