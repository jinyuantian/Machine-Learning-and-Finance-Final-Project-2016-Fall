path = "../ZFB-complete"
outpath = "../ZFB-output"

for i in range(1, 6):
    seq = str(i)
    print('file {}'.format(i))
    fnr = path + "/ZFB-" + seq + ".csv"
    fnw = outpath + "/ZFB-" + seq + "-NoQuote.csv"

    fw = open(fnw, 'w')

    klines = 0
    with open(fnr, 'r') as csvfile:
        for line in csvfile:
            klines += 1
            in_quote = False

            val_first = ""
            for c in line:
                if c == '"':
                    in_quote = not in_quote

                if in_quote:
                    continue

                if c != '"':
                    fw.write(c)

            # fw.write("\n")

    fw.close()

print('Execution finished')
