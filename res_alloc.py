import pandas as pd
from string import ascii_lowercase
import random 
import numpy as np
from itertools import compress
import math

resource = [random.choice(ascii_lowercase) + str(_) for _ in range(100)]
project = [random.choice(ascii_lowercase) + random.choice(ascii_lowercase) +
           str(_) for _ in range(50)]

lang_skill = ["R", "Python", "Scala", "Julia"]
db_skill = ["PSQL", "MySQL", "MongoDB", "Neo4j", "CouchDB"]

random.seed(1311)
resources = pd.DataFrame({
        "name" : resource,
        "skill1" : random.choices(lang_skill, k = 100),
        "skill2" : random.choices(db_skill, k = 100)
            })

projects = pd.DataFrame({
        "project" : project,
        "skill1" : random.choices(lang_skill, k = 50),
        "skill2" : random.choices(db_skill, k = 50)
        })
        
print(resources.head())
print("#########")
print(projects.head())

def schedule_display(sol):
    res = []
    proj = []
    resskill = []
    projskill = []
    slots = []
    # create two slots for each project
    for i in range(len(projects)): slots += [i, i]

    # Loop over resources assignment
    for i in range(len(sol)):
        # get slot
        x = int(sol[i])
        # get resource name
        res.append(resources.name[i])
        # project name
        pr = projects.project[slots[x]]
        # append to project list
        proj.append(pr)
        # get resources skill
        resskill.append(list(resources.iloc[i, 1:]))
        # to get the project skills from the name we need to get the indices
        # where the project is equal to "pr" then slice the projects df
        pr_bool = projects.project == pr
        pr_ind = list(compress(range(len(pr_bool)), pr_bool))
        projskill.append(list(projects.iloc[pr_ind, 1:].values[0]))
        # remove this slot in order not to be filled again
        del slots[x]
    
    res_proj = pd.DataFrame({"Resource" : res, "Project" : proj,
                             "Res_Skill" : resskill,
                             "Proj_Skill" : projskill})
                             
    return res_proj.sort_values("Project")
    
rand_sch = schedule_display([0 for _ in range(len(resources))])
print(rand_sch)

def resproj_cost(sol):
  cost = 0
  # create list a of slots
  slots = []
  for i in range(len(projects)): slots += [i, i]
  
  # loop over each resource
  for i in range(len(sol)):
      x = int(sol[i])
      # get project skills and resources skills
      proj = np.array(projects.iloc[slots[x], 1:])
      res = np.array(resources.iloc[i, 1:])
      # count how many mismatches among skills (0, 1 or 2)
      cost += sum(res != proj)
      
      # remove selected slot
      del slots[x]
    
  return cost
  
def simulated_annealing(domain, costf, temp = 10000.0,
                     cool = 0.95, step = 1):
    # initialize the values randomly
    current_sol = [float(random.randint(domain[i][0], domain[i][1])) for i in range(len(domain))]
    while temp > 0.1:
        # choose one of the indices
        i = random.randint(0, len(domain) - 1)
        
        # choose a direction to change it
        direction = random.randint(- step, step)
        
        # create a new list with one of the values changed
        new_sol = current_sol[:]
        new_sol[i] += direction
        if new_sol[i] < domain[i][0]: new_sol[i] = domain[i][0]
        elif new_sol[i] > domain[i][1]: new_sol[i] = domain[i][1]
        
        # calculate the current cost and the new cost
        current_cost = costf(current_sol)
        new_cost = costf(new_sol)
        #p = pow(math.e, (- new_cost - current_cost) / temp)
        p = math.e ** (( - new_cost - current_cost) / temp)
        
        # is it better, or does it make the probability
        # cutoff?
        if (new_cost < current_cost or random.random() < p):
            current_sol = new_sol
            print(new_cost)
        
        # decrease the temperature
        temp = temp * cool
    return current_sol
    
 
solution = [(0, (len(projects) * 2) - i - 1) for i in range(0, len(projects) * 2)]

# step = 3 to widen the direction of movement and high cool to run the algorithm longer
schedule = simulated_annealing(solution, resproj_cost, step = 3, cool = 0.99)

schedule_df = schedule_display(schedule)
print(schedule_df.head(20))
