import sys

def ResourceHeuristics(heuristic_idx,obs,job_idx):
    if heuristic_idx == 0: #FCFS
        num_exec = obs["num_committable_execs"]-1
    elif heuristic_idx == 1: #DRA
        num_exec = max(1,min(obs["DRA_exec_cap"][job_idx]-obs["exec_supplies"][job_idx], obs["num_committable_execs"]))-1
    elif heuristic_idx == 2: #Fair scheduling
        num_active_jobs = len(obs["DRA_exec_cap"])
        num_exec = max(1,int(obs["num_committable_execs"]/num_active_jobs) - obs["exec_supplies"][job_idx])-1
    else:
        sys.exit("Resource Heuristic idx is not matched to any scheduler")
    #print('num_exec:',num_exec)
    return num_exec