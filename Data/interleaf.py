import sys
import re

simpleFile = sys.argv[1]
islandFile = sys.argv[2]
type = sys.argv[3]

def readbyFour(file):
    chunks = []
    with open(file, "r") as f:
        first_line = f.readline()
        count = 1
        chunk = []
        for line in f.readlines(0):
            chunk += [line.strip("\n").split("\t")]
            if count % 4 == 0:
                chunks.append(chunk)
                chunk = []
            count += 1
    
    return chunks
            

simpleChunks = readbyFour(simpleFile)
islandChunks = readbyFour(islandFile)

def match(simpleChunks, islandChunks):

    matches = []

    for chunkC in simpleChunks:
       for chunkI in islandChunks:
           if chunkC[0][1].split("_", 1)[0] in chunkI[0][1].split("_", 1)[0] and chunkC[0][2] == chunkI[0][2]:
               if chunkC[1][1].split("_", 1)[0] in chunkI[1][1].split("_", 1)[0] and chunkC[1][2] == chunkI[1][2]:
                  if chunkC[2][1].split("_", 1)[0] in chunkI[2][1].split("_", 1)[0] and chunkC[2][2] == chunkI[2][2]:
                      if chunkC[3][1].split("_", 1)[0] in chunkI[3][1].split("_", 1)[0] and chunkC[3][2] == chunkI[3][2]:
                          islandChunks.remove(chunkI)
                          matches.append((chunkC, chunkI))
                          break
    return matches
                
                        
matches = match(simpleChunks, islandChunks)

def WriteMatches():
    with open("eval_" + type + ".tsv", "w+") as f:
        f.write("group\tcondition\ttokens\tcriticalwords\tgrammaticality\n")
        group = 0
        for pair in matches:
            for item in pair[0]:
                f.write(str(group) + "\t" + "\t".join(item) +"\n")
            for item in pair [1]:
                item[0] = re.sub("_","_i_",item[0])
                f.write(str(group) + "\t" + "\t".join(item) +"\n")
            group += 1
    f.close()

WriteMatches()