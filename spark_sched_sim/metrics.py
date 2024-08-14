import numpy as np


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
    durations = []
    for job_id in env.unwrapped.active_job_ids + list(env.unwrapped.completed_job_ids):
        job = env.unwrapped.jobs[job_id]
        t_end = min(job.t_completed, env.unwrapped.wall_time)
        durations += [t_end - job.t_arrival]
        print('job_id:', job_id, '-> arrival:', int(job.t_arrival), ',end:', int(t_end))
    print('durations:', durations)

    for job_id in env.unwrapped.active_job_ids + list(env.unwrapped.completed_job_ids):
        job = env.unwrapped.jobs[job_id]
        t_end = min(job.t_completed, env.unwrapped.wall_time)
        print('job_id:', job_id, '-> start:', int(job.t_arrival), ',end:', int(t_end),
              '(duration:', int(t_end-job.t_arrival),")")
        for stage in job.stages:
            print('  stage_id:', stage.id_)
            for task in stage.completed_tasks:
                print('    task_id:',task.id_, '-> start:', int(task.t_accepted), ',end:', int(task.t_completed),
                      '(duration:', int(task.t_completed-task.t_accepted),")")


def avg_num_jobs(env):
    return sum(job_durations(env)) / env.unwrapped.wall_time


def job_duration_percentiles(env):
    jd = job_durations(env)
    return np.percentile(jd, [25, 50, 75, 100])
