from src.utils import *
from math import exp,log,cos,sqrt,pi


diversity_weights = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Diversity (first 10)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Zero for next
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0   # Zero for last
]

uncertainty_weights = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Zero for first
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Uncertainty (next 10)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0   # Zero for last
]

certainty_weights = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Zero for first
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Zero for next
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0   # Certainty (last 10)
]



class Game:

    def __init__(self):
        self.exploit = lambda B,R : B-R
        self.explore = lambda B, R :  (exp(B) + exp(R))/ (1E-30 + abs(exp(B) - exp(R)))
        self.diverse = lambda B,R : 1 / (abs(B + R) + 1e-30)
        self.a = 0.33
        self.b = 0.33
        self.c = 0.34




    def play(self, i):
        "Sequential model optimization."
        def _ranked(lst:rows) -> rows:
            "Sort `lst` by distance to heaven. Called by `_smo1()`."
            lst = sorted(lst, key = lambda r:d2h(i,r))
            
            return lst

        def _guess(todo:rows, done:rows) -> rows:
            "Divide `done` into `best`,`rest`. Use those to guess the order of unlabelled rows. Called by `_smo1()`."
            cut  = int(.5 + len(done) ** the.N)
            best = clone(i, done[:cut])
            rest = clone(i, done[cut:])
            score = lambda B,R: self.a * self.explore(B,R) + self.b * self.exploit(B,R) + self.c * self.diverse(B,R)
            key  = lambda r: score(loglikes(best, r, len(done), 2),
                                loglikes(rest, r, len(done), 2))


            
            random.shuffle(todo) # optimization: only sort a random subset of todo 
            return  sorted(todo[:the.any], key=key, reverse=True) + todo[the.any:]

            #return sorted(todo,key=key,reverse=True)

        def _smo1(todo:rows, done:rows, most) -> rows:
            "Guess the `top`  unlabeled row, add that to `done`, resort `done`, and repeat"
            itr = 0
            while itr < 30:
                if len(todo) < 3: break
                self.a = uncertainty_weights[itr]
                self.b = certainty_weights[itr]
                self.c = diversity_weights[itr]
                top,*todo = _guess(todo, done)
                most = top if  most ==[] or d2h(i,top) < d2h(i,most) else most
                print(d2h(i,top))
                done += [top]
                done = _ranked(done)
                itr += 1
            return done,most

        random.shuffle(i.rows)
        most = [] # remove any  bias from older runs
        initial = _ranked(i.rows[:the.label])
        done,most = _smo1(i.rows[the.label:],initial, most)
        return done


def smos():
    "try different sample sizes"
    policies = dict(exploit = lambda B,R: B-R,
                    EXPLORE = lambda B,R: (e**B + e**R)/abs(e**B - e**R + 1E-30))
    repeats=20
    
    d = DATA(csv("data/config/SS-X.csv"))
    e = math.exp(1)
    rxs={}
    rxs["baseline"] = SOME(txt=f"baseline,{len(d.rows)}",inits=[d2h(d,row) for row in d.rows])
    the.GuessFaster = True
    for last in [10,20,30,40]:
        the.Last= last
        guess = lambda : clone(d,random.choices(d.rows, k=last),rank=True).rows[0]
        rx=f"random,{last}"
        rxs[rx] = SOME(txt=rx, inits=[d2h(d,guess()) for _ in range(repeats)])
        
        for what,how in  policies.items():
            
            rx=f"{what}/{the.Last}"
            rxs[rx] = SOME(txt=rx)
            for _ in range(repeats):
                btw(".")
                rxs[rx].add(d2h(d,smo(d,how)[0]))
            btw("\n")

    rx = f"Game/30"
    rxs[rx] = SOME(txt = rx)
    for _ in range(repeats):
        result = Game().play(d)
        rxs[rx].add(d2h(d, result[0]))
    report(rxs.values())



smos()