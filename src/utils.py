#!/usr/bin/env python3 -B
# <!-- vim: set ts=2 sw=2 sts=2 et: -->
"""
## Ezr.py
&copy;  2024 Tim Menzies (timm@ieee.org). BSD-2 license


### USAGE:  


python3 ezr.py [OPTIONS]


This code explores multi-objective optimization; i.e. what
predicts for the better goal values? This code also explores
active learning; i.e. how to make predictions after looking at
the fewest number of goal values?


### OPTIONS:  


   -b --buffer int    chunk size, when streaming   = 100
   -B --branch bool   set branch method            = False 
   -d --divide int    half with mean or median     = 0
   -D --Dull   bool   if true, round to cohen's d  = False
   -L --Last   int    max number of labels         = 30 
   -c --cut    float  borderline best:rest         = 0.5 
   -C --Cohen  float  pragmatically small          = 0.35
   -e --eg     str    start up action              = mqs  
   -f --fars   int    number of times to look far  = 20  
   -h --help          show help                    = False 
   -i --iter   int    length of done minus label   = 0
   -k --k      int    low frequency Bayes hack     = 1  
   -l --label  int    initial number of labels     = 4  
   -m --m      int    low frequency Bayes hack     = 2  
   -p --p      int    distance formula exponent    = 2  
   -s --seed   int    random number seed           = 1234567891  
   -S --Stop   int    min size of tree leaves      = 30  
   -t --train  str    training csv file. row1 has names = data/misc/auto93.csv


### Data File Format


Training data consists of csv files where "?" denotes missing values.
Row one  list the columns names, defining the roles of the columns:


- NUMeric column names start with an upper case letter.
- All other columns are SYMbolic.
- Names ending with "+" or "-" are goals to maximize/minimize
- Anything ending in "X" is a column we should ignore.


For example, here is data where the goals are `Lbs-,Acc+,Mpg+`
i.e. we want to minimize car weight and maximize acceleration
and maximize fuel consumption.


    Clndrs   Volume  HpX  Model  origin  Lbs-  Acc+  Mpg+
    -------  ------  ---  -----  ------  ----  ----  ----
     4       90      48   78     2       1985  21.4   40
     4       98      79   76     1       2255  17.7   30
     4       98      68   77     3       2045  18.6   30
     4       79      67   74     2       2000  16     30
     ...
     4      151      85   78     1       2855  17.6   20
     6      168      132  80     3       2910  11.4   30
     8      350      165  72     1       4274  12     10
     8      304      150  73     1       3672  11.5   10


Note that the top rows are
better than the bottom ones (lighter, faster cars that are
more economical).
"""
# todo: labelling via clustering.
# ## Setting-up
# ### Imports
from __future__ import annotations
from typing import Any as any
from typing import List, Dict, Type, Callable, Generator
from fileinput import FileInput as file_or_stdin
from dataclasses import dataclass, field, fields
import datetime
from math import exp,log,cos,sqrt,pi
import re,sys,ast,math,random,inspect
import traceback
from time import time


R = random.random
one = random.choice
#
# ###  Types and Classes
#
# Some misc types:
number  = float  | int   #
atom    = number | bool | str # and sometimes "?"
row     = list[atom]
rows    = list[row]
classes = dict[str,rows] # `str` is the class name


def LIST(): return field(default_factory=list)
def DICT(): return field(default_factory=dict)
#
# NUMs and SYMs are both COLumns. All COLumns count `n` (items seen),
# `at` (their column number) and `txt` (column name).
@dataclass
class COL:
 n   : int = 0
 at  : int = 0
 txt : str = ""


# SYMs tracks  symbol counts  and tracks the `mode` (the most common frequent symbol).
@dataclass
class SYM(COL):
 has  : dict = DICT()
 mode : atom=None
 most : int=0


 def clone(self:SYM): return SYM(at=self.at,txt=self.txt)
# NUMs tracks  `lo,hi` seen so far, as well the `mu` (mean) and `sd` (standard deviation),
# using Welford's algorithm.
@dataclass
class NUM(COL):
 mu : number =  0
 m2 : number =  0
 sd : number =  0
 lo : number =  1E32
 hi : number = -1E32
 goal : number = 1


 def clone(self:NUM): return NUM(at=self.at,txt=self.txt)


 # A minus sign at end of a NUM's name says "this is a column to minimize"
 # (all other goals are to be maximizes).
 def __post_init__(self:NUM) -> None: 
   if  self.txt and self.txt[-1] == "-": self.goal=0
#
# COLS are a factory that reads some `names` from the first
# row , the creates the appropriate columns.
@dataclass
class COLS:
 names: list[str]   # column names
 all  : list[COL] = LIST()  # all NUMS and SYMS
 x    : list[COL] = LIST()  # independent COLums
 y    : list[COL] = LIST()  # dependent COLumns
 klass: COL = None


 # Collect  `all` the COLs as well as the dependent/independent `x`,`y` lists.
 # Upper case names are NUMerics. Anything ending in `+` or `-` is a goal to
 # be maximized of minimized. Anything ending in `X` is ignored.
 def __post_init__(self:COLS) -> None:
   for at,txt in enumerate(self.names):
     a,z = txt[0],txt[-1]
     col = (NUM if a.isupper() else SYM)(at=at, txt=txt)
     self.all.append(col)
     if z != "X":
       (self.y if z in "!+-" else self.x).append(col)
       if z=="!": self.klass = col
       if z=="-": col.goal = 0
#
# DATAs store `rows`, which are summarized in `cols`.
@dataclass
class DATA:
 cols : COLS = None         # summaries of rows
 rows : rows = LIST() # rows


 # Another way to create a DATA is to copy the columns structure of
 # an existing DATA, then maybe load in some rows to that new DATA.
 def clone(self:DATA, rows:rows=[]) -> DATA:
   return DATA().add(self.cols.names).adds(rows)
#
# ### Decorators
# I like how JULIA and CLOS lets you define all your data types
# before anything else. Also, you can group together related methods
# from different classes. I think that really simplifies explaining the
# code. So this `of` decorator lets me
# define methods separately to class definition (and, btw,  it collects a
# documentation strings).
def of(doc):
 def doit(fun):
   fun.__doc__ = doc
   self = inspect.getfullargspec(fun).annotations['self']
   setattr(globals()[self], fun.__name__, fun)
 return doit
#
# ## Methods
# ### Misc
#
@of("Return central tendency of a DATA.")
def mid(self:DATA) -> row:
 return [col.mid() for col in self.cols.all]


@of("Return central tendency of NUMs.")
def mid(self:NUM) -> number: return self.mu


@of("Return central tendency of SYMs.")
def mid(self:SYM) -> number: return self.mode


@of("Return diversity of a NUM.")
def div(self:NUM) -> number: return self.sd


@of("Return diversity of a SYM.")
def div(self:SYM) -> number: return self.ent()


@of("Returns 0..1 for min..max.")
def norm(self:NUM, x) -> number:
 return x if x=="?" else  ((x - self.lo) / (self.hi - self.lo + 1E-32))


@of("Entropy = measure of disorder.")
def ent(self:SYM) -> number:
 return - sum(n/self.n * log(n/self.n,2) for n in self.has.values())


# ### Add
@of("add COL with many values.")
def adds(self:COL,  src) -> COL:
 [self.add(row) for row in src]; return self


@of("add DATA with many values.")
def adds(self:DATA, src) -> DATA:
 [self.add(row) for row in src]; return self


@of("As a side-effect on adding one row (to `rows`), update the column summaries (in `cols`).")
def add(self:DATA,row:row) -> DATA:
 if    self.cols: self.rows += [self.cols.add(row)]
 else: self.cols = COLS(names=row) # for row q
 return self


@of("add all the `x` and `y` cols.")
def add(self:COLS, row:row) -> row:
 [col.add(row[col.at]) for cols in [self.x, self.y] for col in cols]
 return row


@of("If `x` is known, add this COL.")
def add(self:COL, x:any) -> any:
 if x != "?":
   self.n += 1
   self.add1(x)


@of("add symbol counts.")
def add1(self:SYM, x:any) -> any:
 self.has[x] = self.has.get(x,0) + 1
 if self.has[x] > self.most: self.mode, self.most = x, self.has[x]
 return x


@of("add `mu` and `sd` (and `lo` and `hi`). If `x` is a string, coerce to a number.")
def add1(self:NUM, x:any) -> number:
 self.lo  = min(x, self.lo)
 self.hi  = max(x, self.hi)
 d        = x - self.mu
 self.mu += d / self.n
 self.m2 += d * (x -  self.mu)
 self.sd  = 0 if self.n <2 else (self.m2/(self.n-1))**.5
#
# ### Guessing
@of("Guess values at same frequency of `has`.")
def guess(self:SYM) -> any:
 r = R()
 for x,n in self.has.items():
   r -= n/self.n
   if r <= 0: return x
 return self.mode


@of("Guess values with some `mu` and `sd` (using Box-Muller).")
def guess(self:NUM) -> number:
 while True:
   x1 = 2.0 * R() - 1
   x2 = 2.0 * R() - 1
   w = x1*x1 + x2*x2
   if w < 1:
     tmp = self.mu + self.sd * x1 * sqrt((-2*log(w))/w)
     return max(self.lo, min(self.hi, tmp))


@of("Guess a row like the other rows in DATA.")
def guess(self:DATA, fun:Callable=None) -> row:
 fun = fun or (lambda col: col.guess())
 out = ["?" for _ in self.cols.all]
 for col in self.cols.x: out[col.at] = fun(col)
 return out


@of("Guess a value that is more like `self` than  `other`.")
def exploit(self:NUM, other:NUM):
 a = self.like(self.mid())
 b = other.like(self.mid())
 c = (self.n*a - other.n*b)/(self.n + other.n)
 return c,self,self.mid()


@of("Guess a value that is more like `self` than  `other`.")
def exploit(self:SYM, other:SYM):
 priora = self.n/(self.n + other.n)
 priorb = other.n/(self.n + other.n)
 a = self.like(self.mid(),  priora)
 b = other.like(self.mid(), priorb)
 c = a - b
 return c,self,self.mid(),


@of("Guess a row more like `self` than `other`.")
def exploit(self:DATA, other:DATA, top=1000,used=None):
 out = ["?" for _ in self.cols.all]
 for _,col,x in sorted([coli.exploit(colj) for coli,colj in zip(self.cols.x, other.cols.x)],
                      reverse=True,key=nth(0))[:top]:
    out[col.at] = x
    # if used non-nil, keep stats on what is used
    if used != None:
       used[col.at] = used.get(col.at,None) or col.clone()
       used[col.at].add(x)
 return out


@of("Guess a row in between the rows of `self` and `other`.")
def explore(self:DATA, other:DATA):
 out = self.guess()
 for coli,colj in zip(self.cols.x, other.cols.x): out[coli.at] = coli.explore(colj)
 return out


@of("Guess value on the border between `self` and `other`.")
def explore(self:COL, other:COL, n=20):
 n       = (self.n + other.n + 2*the.k)
 pr1,pr2 = (self.n + the.k) / n, (other.n + the.k) / n
 key     = lambda x: abs(self.like(x,pr1) - other.like(x,pr2))
 return min([self.guess() for _ in range(n)], key=key)
#
# ## Distance
@of("Between two values (Aha's algorithm).")
def dist(self:COL, x:any, y:any) -> float:
 return 1 if x==y=="?" else self.dist1(x,y)


@of("Distance between two SYMs.")
def dist1(self:SYM, x:number, y:number) -> float: return x != y


@of("Distance between two NUMs.")
def dist1(self:NUM, x:number, y:number) -> float:
 x, y = self.norm(x), self.norm(y)
 x = x if x !="?" else (1 if y<0.5 else 0)
 y = y if y !="?" else (1 if x<0.5 else 0)
 return abs(x-y)


@of("Euclidean distance between two rows.")
def dist(self:DATA, r1:row, r2:row) -> float:
 n = sum(c.dist(r1[c.at], r2[c.at])**the.p for c in self.cols.x)
 return (n / len(self.cols.x))**(1/the.p)


@of("Sort rows randomly")
def shuffle(self:DATA) -> DATA:
 random.shuffle(self.rows)
 return self


@of("Sort rows by Chebyshev distance.")
def chebyshevs(self:DATA) -> DATA:
 self.rows = sorted(self.rows, key=lambda r: self.chebyshev(r))
 return self


@of("Compute Chebyshev distance of one row to the best `y` values.")
def chebyshev(self:DATA,row:row) -> number:
 return  max(abs(col.goal - col.norm(row[col.at])) for col in self.cols.y)


@of("Sort rows by the Euclidean distance of the goals to heaven.")
def d2hs(self:DATA) -> DATA:
 self.rows = sorted(self.rows, key=lambda r: self.d2h(r))
 return self


@of("Compute euclidean distance of one row to the best `y` values.")
def d2h(self:DATA,row:row) -> number:
 d = sum(abs(c.goal - c.norm(row[c.at]))**2 for c in self.cols.y)
 return (d/len(self.cols.y)) ** (1/the.p)
#
# ### Nearest Neighbor
@of("Sort `rows` by their distance to `row1`'s x values.")
def neighbors(self:DATA, row1:row, rows:rows=None) -> rows:
 return sorted(rows or self.rows, key=lambda row2: self.dist(row1, row2))


@of("Return predictions for `cols` (defaults to klass column).")
def predict(self:DATA, row1:row, rows:rows, cols=None, k=2):
 cols = cols or self.cols.y
 got = {col.at : [] for col in cols}
 for row2 in self.neighbors(row1, rows)[:k]:
   d =  1E-32 + self.dist(row1,row2)
   [got[col.at].append( (d, row2[col.at]) )  for col in cols]
 return {col.at : col.predict( got[col.at] ) for col in cols}
@of("Find weighted sum of numbers (weighted by distance).")
def predict(self:NUM, pairs:list[tuple[float,number]]) -> number:
 ws,tmp = 0,0
 for d,num in pairs:
   w    = 1/d**2
   ws  += w
   tmp += w*num
 return tmp/ws


@of("Sort symbols by votes (voting by distance).")
def predict(self:SYM, pairs:list[tuple[float,any]]) -> number:
 votes = {}
 for d,x in pairs:
   votes[x] = votes.get(x,0) + 1/d**2
 return max(votes, key=votes.get)


 




@of("How much DATA likes a `row`.")
def loglike(self:DATA, r:row, nall:int, nh:int) -> float:
 prior = (len(self.rows) + the.k) / (nall + the.k*nh)
 likes = [c.like(r[c.at], prior) for c in self.cols.x if r[c.at] != "?"]
 return sum(log(x) for x in likes + [prior] if x>0)


@of("How much a SYM likes a value `x`.")
def like(self:SYM, x:any, prior:float) -> float:
 return (self.has.get(x,0) + the.m*prior) / (self.n + the.m)


@of("How much a NUM likes a value `x`.")
def like(self:NUM, x:number, prior=None) -> float:
 v     = self.sd**2 + 1E-30
 nom   = exp(-1*(x - self.mu)**2/(2*v)) + 1E-30
 denom = (2*pi*v) **0.5
 return min(1, nom/(denom + 1E-30))
#
# ### Active Learning
@of("active learning")
def activeLearning(self:DATA, score=lambda B,R: B-R, generate=None, faster=True ):
 def ranked(rows): return self.clone(rows).chebyshevs().rows


 def todos(todo):
   if faster: # Apply our sorting heuristics to just a small buffer at start of "todo"
     # rotate back half of buffer to end of list, fill the gap with later items
      n = the.buffer//2
      return todo[:n] + todo[2*n: 3*n],  todo[3*n:] + todo[n:2*n]
   else: # Apply our sorting heuristics to all of todo.
     return todo,[]


 def guess(todo:rows, done:rows) -> rows:
   cut  = int(.5 + len(done) ** the.cut)
   best = self.clone(done[:cut])
   rest = self.clone(done[cut:])
   a,b  = todos(todo)
   the.iter = len(done) - the.label
   if generate:
     return self.neighbors(generate(best,rest), a) + b
   else:
     key  = lambda r: score(best.loglike(r, len(done), 2), rest.loglike(r, len(done), 2))
     return  sorted(a, key=key, reverse=True) + b


 def loop(todo:rows, done:rows) -> rows:
   while len(todo) > 2 and len(done) < the.Last:
     top,*todo = guess(todo, done)
     done     += [top]
     done      = ranked(done)
   return done
  todo, done = self.rows[the.label:], ranked(self.rows[:the.label])


 return loop(todo, done)
#
# ## Utils


# ### One-Liners


def cdf(x,mu,sd):
  def cdf1(z): return 1 - 0.5*2.718**(-0.717*z - 0.416*z*z)
  z = (x - mu) / sd
  return cdf1(z) if z >= 0 else 1 - cdf1(-z)


# non parametric mid and div
def medianSd(a: list[number]) -> tuple[number,number]:
 a = sorted(a)
 return a[int(0.5*len(a))], (a[int(0.9*len(a))] - a[int(0.1*len(a))])


# Return a function that returns the `n`-th idem.
def nth(n): return lambda a:a[n]


# Rounding off
def r2(x): return round(x,2)
def r3(x): return round(x,3)


# Pring to standard error
def dot(s="."): print(s, file=sys.stderr, flush=True, end="")


# Timing
def timing(fun) -> number:
 start = time()
 fun()
 return time() - start


# M-by-N cross val
def xval(lst:list, m:int=5, n:int=5, some:int=10**6) -> Generator[rows,rows]:
 for _ in range(m):
   random.shuffle(lst)
   for n1 in range (n):
     lo = len(lst)/n * n1
     hi = len(lst)/n * (n1+1)
     train, test = [],[]
     for i,x in enumerate(lst):
       (test if i >= lo and i < hi else train).append(x)
     train = random.choices(train, k=min(len(train),some))
     test = random.choices(test, k=min(len(test),some))
     yield train,test


# ### Strings to Things


def coerce(s:str) -> atom:
 "Coerces strings to atoms."
 try: return ast.literal_eval(s)
 except Exception:  return s


def csv(file) -> Generator[row]:
 infile = sys.stdin if file=="-" else open(file)
 with infile as src:
   for line in src:
     line = re.sub(r'([\n\t\r ]|#.*)', '', line)
     if line: yield [coerce(s.strip()) for s in line.split(",")]


# ### Settings and CLI
class SETTINGS:
 def __init__(self,s:str) -> None:
   "Make one slot for any line  `--slot ... = value`"
   self._help = s
   want = r"\n\s*-\w+\s*--(\w+).*=\s*(\S+)"
   for m in re.finditer(want,s): self.__dict__[m[1]] = coerce(m[2])
   self.sideEffects()


 def __repr__(self) -> str:
   "hide secret slots (those starting with '_'"
   return str({k:v for k,v in self.__dict__.items() if k[0] != "_"})


 def cli(self):
   "Update slots from command-line"
   d = self.__dict__
   for k,v in d.items():
     v = str(v)
     for c,arg in enumerate(sys.argv):
       after = sys.argv[c+1] if c < len(sys.argv) - 1 else ""
       if arg in ["-"+k[0], "--"+k]:
         d[k] = coerce("False" if v=="True" else ("True" if v=="False" else after))
   self.sideEffects()


 def sideEffects(self):
   "Run side-effects."
   d = self.__dict__
   random.seed(d.get("seed",1))
   if d.get("help",False):
     sys.exit(print(self._help))
#


class o:
 __init__ = lambda i,**d : i.__dict__.update(d)
 __repr__ = lambda i     : i.__class__.__name__+"("+dict2str(i.__dict__)+")"


The = o(
 seed  = 1234567891,
 round = 2,
 stats = o(cohen=0.35,
           cliffs=0.195, #border between small=.11 and medium=.28
           bootstraps=512,
           confidence=0.05))




class SOME:
   "Non-parametric statistics using reservoir sampling."
   def __init__(i, inits=[], txt="", max=512):
     "Start stats. Maybe initialized with `inits`. Keep no more than `max` numbers."
     i.txt,i.max,i.lo, i.hi  = txt,max, 1E30, -1E30
     i.rank,i.n,i._has,i.ok = 0,0,[],True
     i.adds(inits) 


   def __repr__(i):
     "Print the reservoir sampling."
     return  'SOME('+str(dict(txt=i.txt,rank="i.rank",n=i.n,all=len(i.has()),ok=i.ok))+")"


   def adds(i,a): 
     "Handle multiple nests samples."
     for b in a:
       if   isinstance(b,(list,tuple)): [i.adds(c) for c in b] 
       elif isinstance(b,SOME):         [i.add(c) for c in b.has()]
       else: i.add(b)


   def add(i,x): 
     i.n += 1
     i.lo = min(x,i.lo)
     i.hi = max(x,i.hi)
     now  = len(i._has)
     if   now < i.max   : i.ok=False; i._has += [x]
     elif R() <= now/i.n: i.ok=False; i._has[ int(R() * now) ]


   def __eq__(i,j):
     "True if all of cohen/cliffs/bootstrap say you are the same."
     return  i.cliffs(j) and i.bootstrap(j) ## ordered slowest to fastest


   def has(i) :
     "Return the numbers, sorted."
     if not i.ok: i._has.sort()
     i.ok=True
     return i._has


   def mid(i):
     "Return the middle of the distribution."
     l = i.has(); return l[len(l)//2]


   def div(i):
      "Return the deviance from the middle."
      l = i.has(); return (l[9*len(l)//10] - l[len(l)//10])/2.56


   def pooledSd(i,j):
     "Return a measure of the combined standard deviation."
     sd1, sd2 = i.div(), j.div()
     return (((i.n - 1)*sd1 * sd1 + (j.n-1)*sd2 * sd2) / (i.n + j.n-2))**.5


   def norm(i, n):
     "Normalize `n` to the range 0..1 for min..max"
     return (n-i.lo)/(i.hi - i.lo + 1E-30)


   def bar(i, some, fmt="%8.3f", word="%-50s ", width=50):
     "Pretty print `some.has`."
     has = some.has()
     out = [' '] * width
     cap = lambda x: 1 if x > 1 else (0 if x<0 else x)
     #pos = lambda x: int(width * cap(i.norm(x)))
     pos = lambda x: min(int(width * cap(i.norm(x))), width - 1)
     [a, b, c, d, e]  = [has[int(len(has)*x)] for x in [0.1,0.3,0.5,0.7,0.9]]
     [na,nb,nc,nd,ne] = [pos(x) for x in [a,b,c,d,e]]
     for j in range(na,nb): out[j] = "-"
     for j in range(nd,ne): out[j] = "-"
     out[width//2] = "|"
     out[nc] = "*"
     #return ', '.join(["%2d" % some.rank, word % some.txt, fmt%c, fmt%(d-b), ''.join(out)])
     return ', '.join(["%2d" % some.rank, some.txt.rjust(25," "), fmt%c, fmt%(d-b), ''.join(out),
                       ', '.join([fmt % x for x in
                                     [a,b,c,d,e]])])


   def delta(i,j):
     "Report distance between two SOMEs, modulated in terms of the standard deviation."
     return abs(i.mid() - j.mid()) / ((i.div()**2/i.n + j.div()**2/j.n)**.5 + 1E-30)


   def cohen(i,j):
     return abs( i.mid() - j.mid() ) < The.stats.cohen * i.pooledSd(j)


   def cliffs(i,j, dull=None):
     """non-parametric effect size. threshold is border between small=.11 and medium=.28
     from Table1 of  https://doi.org/10.3102/10769986025002101"""
     n,lt,gt = 0,0,0
     for x1 in i.has():
       for y1 in j.has():
         n += 1
         if x1 > y1: gt += 1
         if x1 < y1: lt += 1
     return abs(lt - gt)/n  < (dull or The.stats.cliffs or 0.197) 


   def  bootstrap(i,j,confidence=None,bootstraps=None):
     """non-parametric significance test From Introduction to Bootstrap,
       Efron and Tibshirani, 1993, chapter 20. https://doi.org/10.1201/9780429246593"""
     y0,z0  = i.has(), j.has()
     x,y,z  = SOME(inits=y0+z0), SOME(inits=y0), SOME(inits=z0)
     delta0 = y.delta(z)
     yhat   = [y1 - y.mid() + x.mid() for y1 in y0]
     zhat   = [z1 - z.mid() + x.mid() for z1 in z0]
     pull   = lambda l:SOME(random.choices(l, k=len(l)))
     samples= bootstraps or The.stats.bootstraps or 512
     n      = sum(pull(yhat).delta(pull(zhat)) > delta0  for _ in range(samples))
     return n / samples >= (confidence or The.stats.confidence or 0.05)


# ---------------------------------------------------------------------------------------
def sk(somes,epsilon=0.01):
 "Sort nums on mid. give adjacent nums the same rank if they are statistically the same"
 def sk1(somes, rank, cut=None):
   most, b4 = -1, SOME(somes)
   for j in range(1,len(somes)):
     lhs = SOME(somes[:j])
     rhs = SOME(somes[j:])
     tmp = (lhs.n*abs(lhs.mid() - b4.mid()) + rhs.n*abs(rhs.mid() - b4.mid())) / b4.n
     if tmp > most and (somes[j].mid() - somes[j-1].mid()) > epsilon:
        most,cut = tmp,j
   if cut:
     some1,some2 = SOME(somes[:cut]), SOME(somes[cut:])
     if True: #not some1.cohen(some2): # and abs(some1.div() - some2.div()) > 0.0001:
       if some1 != some2:
         rank = sk1(somes[:cut], rank) + 1
         rank = sk1(somes[cut:], rank)
         return rank
   for some in somes: some.rank = rank
   return rank
 somes = sorted(somes, key=lambda some: some.mid()) #lambda some : some.mid())
 sk1(somes,0)
 return somes


def file2somes(file):
 "Reads text file into a list of `SOMEs`."
 def asNum(s):
   try: return float(s)
   except Exception: return s
 somes=[]
 with open(file) as fp:
   for word in [asNum(x) for s in fp.readlines() for x in s.split()]:
     if isinstance(word,str): some = SOME(txt=word); somes.append(some)
     else                   : some.add(word)   
 return somes


def bars(somes, width=40,epsilon=0.01,fmt="%5.2f"):
 "Prints multiple `somes` on the same scale."
 all = SOME(somes)
 last = None
 for some in sk(some, epsilon):
   if some.rank != last: print("#")
   last=some.rank
   print(all.bar(some.has(), width=width, word="%20s", fmt=fmt))
def report(somes,epsilon=0.01,fmt="%5.2f"):
 all = SOME(somes)
 last = None
 #print(SOME(inits=[x for some in somes for x in some._has]).div()*the.stats.cohen)
 for some in sk(somes,epsilon):
   if some.rank != last: print("#")
   last=some.rank
   print(all.bar(some,width=40,word="%20s", fmt=fmt))


# ## Tests
class egs:
 def all():
  for s in dir(egs):
    if s[0] != "_" and s != "all":
       print("\n---------------------------",s,"-------------------------------")
       random.seed(the.seed)
       try:
           getattr(egs,s)()
       except Exception:
           print("FAIL!!!!!! ",s,Exception)


  def policies():
   def normalized_exp(k, n, shift):
       exp_values = [math.exp(0.25 * j) for j in range(n)]
       min_exp, max_exp = min(exp_values), max(exp_values)
       return (math.exp(0.25 * k) - min_exp) / (max_exp - min_exp) + shift
  
   def exploit(B, R):
       return B
  
   def explore(B, R):
       return (B + R) / (abs(B - R) + 10**-30)
  
   scoring_policies = [('exploit', lambda B, R,: exploit(B, R)),
                       ('explore', lambda B, R: explore(B, R)),
                       ('b2', lambda B, R: (B**2) / (R + 10**-30)),
                       ('Random', lambda B, R: random.random()),
                       ('SimAnneal', lambda B, R: ((exp(B) + 1) ** normalized_exp(the.iter, the.Last, 1) + (exp(R) + 1)) / (abs(exp(B) - exp(R)) + 10**-30)),
                       ('ExpProgressive', lambda B, R: normalized_exp(the.iter, the.Last, 0) * exploit(B,R) + (1 - normalized_exp(the.iter, the.Last, 0)) * explore(B,R))]
  
   print(the.train,  flush=True, file=sys.stderr)
   print("\n"+the.train)
   repeats  = 20
   d        = DATA().adds(csv(the.train))
   b4       = [d.chebyshev(row) for row in d.rows]
   asIs,div = medianSd(b4)
   rnd      = lambda z: z


   print(f"asIs\t: {asIs:.3f}")
   print(f"div\t: {div:.3f}")
   print(f"rows\t: {len(d.rows)}")
   print(f"xcols\t: {len(d.cols.x)}")
   print(f"ycols\t: {len(d.cols.y)}\n")


   somes = [stats.SOME(b4,f"asIs,{len(d.rows)}")]


   for what,how in scoring_policies:
     for the.Last in [20, 30, 40]:
       start = time()
       result = [rnd(d.chebyshev(d.shuffle().activeLearning(score=how)[0]))
               for _ in range(repeats)]
       print(f"{what}.{the.Last}: {(time() - start) /repeats:.2f} secs")
       somes +=   [stats.SOME(result,    f"{what} ,{the.Last}")]


   stats.report(somes, 0.01)








 def branch():
   scoring_policies = [('exploit', lambda B, R,: B - R),
                       ('explore', lambda B, R :  (exp(B) + exp(R))/ (1E-30 + abs(exp(B) - exp(R))))]
  
   print(the.train,  flush=True, file=sys.stderr)
   print("\n"+the.train)
   repeats  = 20
   d        = DATA().adds(csv(the.train))
   b4       = [d.chebyshev(row) for row in d.rows]
   asIs,div = medianSd(b4)
   rnd      = lambda z: z


   print(f"asIs\t: {asIs:.3f}")
   print(f"div\t: {div:.3f}")
   print(f"rows\t: {len(d.rows)}")
   print(f"xcols\t: {len(d.cols.x)}")
   print(f"ycols\t: {len(d.cols.y)}\n")


   somes = [SOME(b4,f"asIs,{len(d.rows)}")]


   for what,how in scoring_policies:
     for the.Last in [0,20, 30, 40]:
       for the.branch in [False, True]:
         start = time()
         result = []
         runs = 0
         for _ in range(repeats):
            tmp=d.shuffle().activeLearning(score=how)
            runs += len(tmp)
            result += [rnd(d.chebyshev(tmp[0]))]


         pre=f"{what}/b={the.branch}" if the.Last >0 else "rrp"
         tag = f"{pre},{int(runs/repeats)}"
         print(tag, f": {(time() - start) /repeats:.2f} secs")
         somes +=   [SOME(result,    tag)]


   report(somes, 0.01)


the = SETTINGS(__doc__)
if __name__ == "__main__" and len(sys.argv)> 1:
 the.cli()
 random.seed(the.seed)
 getattr(egs, the.eg, lambda : print(f"ezr: [{the.eg}] unknown."))()






R=random.random



