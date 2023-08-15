import numpy as np
import scipy as sp # may be useful to compute probabilities
import time # may be useful to check the execution time of some function


"""
Starting and ending locations (indices) of red and green exons in the reference sequence - Begins

1. Red Exon Locations
"""


RedExonPos = np.array([
    [149249757, 149249868], # R1
    [149256127, 149256423], # R2
    [149258412, 149258580], # R3
    [149260048, 149260213], # R4
    [149261768, 149262007], # R5
    [149264290, 149264400]  # R6
    ])
"""
2. Green Exon Locations
"""
GreenExonPos = np.array([
    [149288166, 149288277], # G1
    [149293258, 149293554], # G2
    [149295542, 149295710], # G3
    [149297178, 149297343], # G4
    [149298898, 149299137], # G5
    [149301420, 149301530]  # G6
    ])
"""
Starting and ending locations (indices) of red and green exons in the reference sequence - Ends
"""    




def loadLastCol(filename):
    """
    Input: Path of the file corresponding the last column (BWT).
    Output: The last column (BWT) in string format.
    """    
    global RankMatrix
    global firstOccurence

    def initialize_firstOccurence_and_Rank_matrix(LastCol):
        counts = {'A':0,'C':0,'G':0,'T':0}
        n = len(LastCol)
        RankMatrix = {'A': np.zeros(n), 'C': np.zeros(n), 'G': np.zeros(n), 'T': np.zeros(n)}
        i=0
        for item in LastCol:
            for id in ['A','C','G','T']:
                
                if id!=item:
                    RankMatrix[id][i]=  counts[id]
                else:
                    RankMatrix[item][i]=  counts[item] + 1
                    
            if item!='$':
                counts[item]+=1
            i+=1


        count=0
        for item in counts:
            count = count + counts[item]
            counts[item] = count

        firstOccurence= {'A':0,'C':counts['A'],'G':counts['C'],'T':counts['G']}
        return firstOccurence,RankMatrix

    ##### function begins   ###############
    
    x = open(filename)
    LastCol = x.read().replace('\n', '')

    firstOccurence,RankMatrix = initialize_firstOccurence_and_Rank_matrix(LastCol)
    print("FirstOccurence and Rank Matrix Created")
    ##### function ends   ###############

    return LastCol #string data type
    
    
    

def loadRefSeq(filename):
    """
    Input: Path of the file containing the reference sequence.
    Output: The reference sequence in string format.
    """
    # function body - Begins
    x = open(filename)
    RefSeq = x.read().replace('\n', '').replace(">chrX", '')
    # function body - Ends
    return RefSeq # string data type

def loadReads(filename):
    """
    Input: Path of the file containing all the reads.
    Output: A list containing all the reads.
    """
    # function body - Begins
    x = open(filename)
    Reads = x.read().split('\n')
    # function body - Ends
    return Reads[:-1] # list of strings

def loadMapToRefSeq(filename):
    """
    Input: Path of the file containing mapping of the first column to the reference sequence.
    Output: numpy integer array containing the mapping.
    """
    # function body - Begins
    x = open(filename)

    list_of_strings = x.read().split("\n")
    MapToRefSeq = list_of_strings[:-1]
    print("########### Map Loaded #########")
    print("########### Reads Scanning Started #########")
    # function body - Ends
    return MapToRefSeq # numpy integer array


def MatchReadToLoc(read):
    """
    Input: a read (string)
    Output: list of potential locations at which the read may match the reference sequence. 
    """
    # function body - Begins
    positions = []
    
    def getMisMatchList(top,bottom):
        syList = []
        for symbol in ['A','C','G','T']:
            if(RankMatrix[symbol][top-1]==RankMatrix[symbol][bottom]): continue
            syList.append(symbol)
        return syList
        
    def matchRead(read,top,bottom,mismatch):        
        while(top<=bottom):
            if read!='':
                symbol = read[-1]
                read = read[:-1]
                
                if symbol=='N': symbol='A'
                    
                if(RankMatrix[symbol][top-1]==RankMatrix[symbol][bottom] and top>0):  
                    if mismatch<2 :
                        mismatch+=1
                        syList = getMisMatchList(top,bottom)    
                        positions=[]
                        for symbol in syList:
                            top = int(firstOccurence[symbol] + RankMatrix[symbol][top]) - (1 if RankMatrix[symbol][top] else 0) 
                            bottom = int(firstOccurence[symbol] + RankMatrix[symbol][bottom]) - (1 if RankMatrix[symbol][bottom] else 0)
                            positions+=matchRead(read,top,bottom,mismatch)
                        return positions
                    else:
                        return []
                
                else:
                    top = int(firstOccurence[symbol] + RankMatrix[symbol][top]) - (1 if RankMatrix[symbol][top] else 0) 
                    bottom = int(firstOccurence[symbol] + RankMatrix[symbol][bottom]) - (1 if RankMatrix[symbol][bottom] else 0)
            else:
                res = []
                for item in list(range(top+1,bottom+1)):
                    res.append(int(Map[item]))
                return res
        
        return []
        
    
    rev_comp_read = ''
    comp = {'A':'T','C':'G','G':'C','T':'A'}
    for char in read[::-1]:
        if char == 'N':
            rev_comp_read += 'T'
            continue
        rev_comp_read += comp[char]
    
    positions = matchRead(read,0,len(LastCol)-1,0)
    positions += matchRead(rev_comp_read,0,len(LastCol)-1,0)


    return positions # list of potential locations at which the read may match the reference sequence.


def WhichExon(positions):
    """
    Input: list of potential locations at which the read may match the reference sequence.
    Output: Update(increment) to the counts of the 12 exons
    """
    
    resR = [0]*6
    resG = [0]*6
    flag= [0]*6
    
    for pos in positions:             
        for i in range(6):
            if pos >= RedExonPos[i][0] and pos <= RedExonPos[i][1]:   #update if any Red Exon location matches
                resR[i]+=1
                flag[i]+=1
            if pos >= GreenExonPos[i][0] and pos <= GreenExonPos[i][1]: #update if any Red Exon location matche
                resG[i]+=1
                flag[i]+=1
    
    for i in range(6):                 #used to update values that are updated for both green and red for a particular exon 
        if flag[i]>1 :                 # for both red and green correspondingly
            resR[i] -= ((flag[i]//2)*1/2)
            resG[i] -= ((flag[i]//2)*1/2)

    resR = resR + resG
    
    return np.array(resR)

def ComputeProb(ExonMatchCounts):
    """
    Input: The counts for each exon
    Output: Probabilities of each of the four configurations (a list of four real numbers)
    """
    print("Exons Counts: ",ExonMatchCounts)
    
    
    # function body - Begins
    P0 = ExonMatchCounts[1]/ExonMatchCounts[7]
    P1 = ExonMatchCounts[2]/ExonMatchCounts[8]
    P2 = ExonMatchCounts[3]/ExonMatchCounts[9]
    P3 = ExonMatchCounts[4]/ExonMatchCounts[10]
    # function body - ends
    print("Probablity of Given Exons is as follows: ",[P0, P1, P2, P3])
    return [P0, P1, P2, P3]

def BestMatch(ListProb):
    """
    Input: Probabilities of each of the four configurations (a list of four real numbers)
    Output: Most likely configuration (an integer). Refer to lecture slides
    """
    
    def euclidean_distance(a,b):
        sum=0
        for i in range(len(a)):
            sum += (a[i]-b[i])**2
        
        return np.sqrt(sum)

    # function body - Begins
    cfg = [[0.5,0.5,0.5,0.5],[1,1,0,0],[0.33,0.33,1,1],[0.33,0.33,0.33,1]]
    MostLikelyConfiguration = 0
    mn = 100

    for i in range(4):
        ed = euclidean_distance(ListProb,cfg[i])
        if ed<mn:
            mn = ed
            MostLikelyConfiguration =i

    return MostLikelyConfiguration # it holds 0, 1, 2, or 3

if __name__ == "__main__":
    # load all the data files
    LastCol = loadLastCol("../data/chrX_last_col.txt") # loads the last column
    RefSeq = loadRefSeq("../data/chrX.fa") # loads the reference sequence
    Reads = loadReads("../data/reads") # loads the reads
    Map = loadMapToRefSeq("../data/chrX_map.txt") # loads the mapping to the reference sequence

    # run the functions
    ExonMatchCounts = np.zeros(12) # initialize the counts for exons

    for read in Reads: # update the counts for exons
        positions = MatchReadToLoc(read) # get the list of potential match locations
        ExonMatchCounts += WhichExon(positions) # update the counts of exons, if applicable
        
    ListProb = ComputeProb(ExonMatchCounts) # compute probabilities of each of the four configurations
    MostLikely = BestMatch(ListProb) # find the most likely configuration
    print("Configuration %d is the best match"%MostLikely)
