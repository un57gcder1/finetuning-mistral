import json

FILES = ["ja1-train.txt", "ja1-valid.txt"]
NEW = ["ja1-train.json", "ja1-valid.json"]
for i in range(len(FILES)):
    f = open(FILES[i], 'r')
    n = open(NEW[i], 'w')
    line = f.readline()
    while line:
        line = "<s> " + line + " </s>"
        nS = {"text": line}
        nS = json.dumps(nS)
        nS += "\n"
        n.write(nS)
        line = f.readline()
    print("Completed " + FILES[i] + " ==> " + NEW[i] + ".")
