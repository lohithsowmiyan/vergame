from src.utils import *

class Game:

    def __init__(self, rows):
        self.explore = lambda B,R : B-R 
        self.exploit = lambda B, R :  (exp(B) + exp(R))/ (1E-30 + abs(exp(B) - exp(R)))
        self.diverse = lambda B,R : 1 / abs(B + R + 1e-30)
        self.a = 0.33
        self.b = 0.33
        self.c = 0.34
        self.rows = rows



    def play(self):
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
                score = lambda B, R : self.a * self.explore(B,R) + self.b * self.exploit(B,R) + self.c * self.diverse(B,R)
                key  = lambda r : score(best.loglike(r, len(done), 2), rest.loglike(r, len(done), 2))
                return  sorted(a, key=key, reverse=True) + b


        def loop(todo:rows, done:rows) -> rows:
            while len(todo) > 2 and len(done) < the.Last:

                self.a = int(input("Enter weight for explore"))
                self.b = int(input("Enter weight for exploit"))
                self.b = int(input("Enter weight for diverse"))

                top,*todo = guess(todo, done)
                done     += [top]
                done      = ranked(done)
            return done
        todo, done = self.rows[the.label:], ranked(self.rows[:the.label])


        return loop(todo, done)


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
        start = time()
        result = []
        runs = 0
        for _ in range(repeats):
            tmp=d.shuffle().activeLearning(score=how)
            runs += len(tmp)
            result += [rnd(d.chebyshev(tmp[0]))]


        pre=f"{what}" if the.Last >0 else "rrp"
        tag = f"{pre},{int(runs/repeats)}"
        print(tag, f": {(time() - start) /repeats:.2f} secs")
        somes +=   [SOME(result,    tag)]
        
   rslt = []
   the.Last = 30
   for _ in range(repeats):
        game = Game(d.rows)
        rslt += [rnd(d.chebyshev(game.play()[0]))]

   pre = f"Game/30"
   tag = pre
   somes += [SOME(result,tag)]





   report(somes, 0.01)

branch()