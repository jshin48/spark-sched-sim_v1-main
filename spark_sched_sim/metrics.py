import numpy as np
import csv
from datetime import datetime

def job_durations(env):
    durations = []
    for job_id in env.unwrapped.active_job_ids + list(env.unwrapped.completed_job_ids):
        job = env.unwrapped.jobs[job_id]
        t_end = min(job.t_completed, env.unwrapped.wall_time)
        durations += [t_end - job.t_arrival]
    return durations


def avg_job_duration(env):
    return np.mean(job_durations(env))

def print_task_job_time(env):
    current_time = datetime.now().strftime("%m%d_%H%M%S")

    f = open('./results/detail/'+str(env.agent_cls)+str(current_time)+'.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    header = ['job_id','stage_id','task_id','job_arrival_t','task_start_t','task_end_t','task_dur','job_end_t','job_dur']
    writer.writerow(header)
    for job_id in env.unwrapped.active_job_ids + list(env.unwrapped.completed_job_ids):
        job = env.unwrapped.jobs[job_id]
        t_end = min(job.t_completed, env.unwrapped.wall_time)
        for stage in job.stages:
            for task in stage.completed_tasks:
                row = [job_id,stage.id_,task.id_,job.t_arrival,task.t_accepted,task.t_completed,
                      int(task.t_completed-task.t_accepted), t_end,int(t_end-job.t_arrival)]
                writer.writerow(row)


def avg_num_jobs(env):
    return sum(job_durations(env)) / env.unwrapped.wall_time


def job_duration_percentiles(env):
    jd = job_durations(env)
    return np.percentile(jd, [25, 50, 75, 100])
