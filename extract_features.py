import json
import os
import re
import numpy as np

sourcedir = './GraphProt_CLIP_sequences/'
fnums=168+96+2
files=[]
n2num={'a':0,'u':1,'c':2,'g':3}

def kmer2idx(s):
    idx=0
    for i in range(len(s)):
        if s[i]=='a' or s[i]=='u' or s[i]=='c' or s[i]=='g':
            idx=idx+(n2num[s[i]]<<(2*i))
        else:
            return -1
    return idx

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.fa'):
                fullname = f
                files.append(fullname)

def getBOW(seq):
    list1=[0]*4
    list2=[0]*16
    list3=[0]*64
    if len(seq)!=0:
        for nb in seq:
            list1[kmer2idx(nb)]+=1
        list1=list(map(lambda x:x/len(seq), list1))
        for i in range(len(seq)-2+1):
            subseq=seq[i:i+2]
            idx=kmer2idx(subseq)
            list2[idx]+=1
        if (len(seq)-1)>0:
            list2=list(map(lambda x:x/(len(seq)-1), list2))
        for i in range(len(seq)-3+1):
            subseq=seq[i:i+3]
            idx=kmer2idx(subseq)
            list3[idx]+=1
        if (len(seq)-2)>0:
            list3=list(map(lambda x:x/(len(seq)-2), list3))
    return list1+list2+list3

def getRelation(seq):
    stride=[4,5,6]
    list1=[0]*48
    if len(seq)!=0:
        for i in range(len(stride)):
            m=stride[i]
            for j in range(len(seq)-m):
                subseq=seq[j]+seq[j+m]
                idx=16*i+kmer2idx(subseq)
                list1[idx]+=1
            if (len(seq)-m)>0:
                for j in range(i*16,(i+1)*16):
                    list1[j]=list1[j]/(len(seq)-m)
    return list1

def getOthers(seq):
    list1=[]
    if len(seq)%3==0:
        list1.append(1)
    else:
        list1.append(0)
    if 'uag' in seq or 'uaa' in seq or 'uga' in seq:
        list1.append(1)
    else:
        list1.append(0)
    return list1
         
def extractFeatures(filename):
    dic={}
    path=sourcedir+filename
    name=''
    flist=[]
    cnt=0
    for line in open(path, "r"):
        if line[0] == '>':
            name=line.strip('\n').strip('\r')
            cnt=cnt+1
            if cnt%500==0:
                print(cnt)
        else:
            if ('n' in line or 'N' in line):
                dic[name,[0]*fnums]
                continue
            #print(filename,' ',name)
            line=line.strip('\n').strip('\r')
            line=line.replace('T','U')
            line=line.replace('t','u')
            #if cnt>200 and cnt<300:
            #    print(cnt,":")
            searchObj = re.search( r'([aucg]*)([AUCG]*)([aucg]*)', line)
            upstream = searchObj.group(1)
            bindsite = searchObj.group(2).lower()
            downstream = searchObj.group(3)
            
            ulist1=getBOW(upstream)
            ulist2=getRelation(upstream)
            ulist3=getOthers(upstream)
            blist1=getBOW(bindsite)
            blist2=getRelation(bindsite)
            blist3=getOthers(bindsite)
            dlist1=getBOW(downstream)
            dlist2=getRelation(downstream)
            dlist3=getOthers(downstream)
            list1=[]
            list2=[]
            list3=[]
            for i in range(len(ulist1)):
                list1.append((ulist1[i]+dlist1[i])/2)
            for i in range(len(ulist2)):
                list2.append((ulist2[i]+dlist2[i])/2)
            for i in range(len(ulist3)):
                list3.append((ulist3[i]+dlist3[i])/2)
            flist=list1+blist1+list2+blist2+list3+blist3
            dic[name]=flist
    with open(filename.rstrip('.fa')+'.txt', 'w') as f:
        json.dump(dic,f)

if __name__ == '__main__':
    findAllFile(sourcedir)
    for item in files:
        print(item)
        extractFeatures(item) 
    
