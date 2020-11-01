import os

file = './sequence.fasta'
files=[]
seq=''
window=[33,39,35,27]
step=[8,10,9,7]
prefix='>SARS-CoV-2;'
save=['ALKBH5_Baltz2012.SARS-CoV-2.txt','ICLIP_HNRNPC.SARS-CoV-2.txt','PARCLIP_MOV10.SARS-CoV-2.txt','PTBv1.SARS-CoV-2.txt']

def getSeq():
    global seq
    for line in open(file, "r"):
        if line[0] == '>':
            continue
        else:
            line=line.strip('\n').strip('\r')
            seq+=line.replace('T','U')

def generateTestfile(path, index):
    f=open(save[index],'w')
    title=''
    subseq=''

    for i in range(150,len(seq)-150*2-window[index],step[index]):
        title=prefix+str(i)+','+str(i+window[index])+','+'\n'
        f.write(title)
        subseq=seq[i-150:i].lower()+seq[i:i+window[index]]+seq[i+window[index]:i+window[index]+150].lower()+'\n'
        f.write(subseq)
    f.close()

if __name__ == '__main__':
    getSeq()
    for i in range(4):
        generateTestfile(file,i)


