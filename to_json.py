import json

DELIMITERS = ["", " ", "\n"]
FILES = ["ja1-train.txt", "ja1-valid.txt"]
NEW = ["ja1-train.json", "ja1-valid.json"]

def delimiter(s):
    for d in DELIMITERS:
        if d == s:
            return True
    return False

for i in range(len(FILES)):
    f = open(FILES[i], 'r')
    n = open(NEW[i], 'w')
    line = f.readline()
    total = ""
    while line:
        total += line
        if (delimiter(line) and total == ""):
            continue
        elif (delimiter(line) and total not in DELIMITERS):
            total = total.replace("\n", " ")
            nS = {"text": total}
            nS = json.dumps(nS)
            nS += "\n"
            n.write(nS)
            total = ""
        elif (total in DELIMITERS):
            total = ""
        line = f.readline()
    print("Completed " + FILES[i] + " ==> " + NEW[i] + ".")
