import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from components import Job, Stage

def parse_jobs_from_obs(obs):
    # Use helper function to convert obs into Job/Stage objects (only for scheduling algorithms)
    jobs = []
    dag_ptr = obs["dag_ptr"]
    nodes = obs["dag_batch"].nodes  # numpy array
    stage_mask = obs["stage_mask"]
    num_jobs = len(dag_ptr) - 1

    for job_idx in range(num_jobs):
        stage_idx_start = dag_ptr[job_idx]
        stage_idx_end = dag_ptr[job_idx + 1]

        stages = []
        for stage_idx in range(stage_idx_start, stage_idx_end):
            num_tasks = int(nodes[stage_idx][3])
            is_schedulable = bool(stage_mask[stage_idx])
            cpt = float(nodes[stage_idx][5])
            num_children = int(nodes[stage_idx][6])

            stage = Stage(
                id=stage_idx - stage_idx_start,  # stage id within job
                job_id=job_idx,
                num_tasks=num_tasks,
                task_duration = 0,#task_duration is not used in this context
                cpt=cpt,
                num_children=num_children
            )

            stage.is_schedulable = is_schedulable
            stage.remaining_work = float(nodes[stage_idx][4])
            stages.append(stage)

        job = Job(
            id_=job_idx,
            stages=stages,
            dag=None,  # optional - not used in this context
            t_arrival=0,  # optional
            cpt=max(s.cpt for s in stages)  # optional
        )
        # Manually set frontier stages
        job.frontier_stages = set(s for s in stages if s.is_schedulable)

        # Total work of all active stages
        job.total_remaining_work = sum(stage.remaining_work for stage in stages)
        jobs.append(job)

    return jobs
