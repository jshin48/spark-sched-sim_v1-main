import argparse
import ast

parser = argparse.ArgumentParser(description='HyperHeuristic')

# -- Basic --
parser.add_argument('--num_executors', type=int, default=20)
parser.add_argument('--job_arrival_cap', type=int, default=100)
parser.add_argument('--job_arrival_rate', type=float, default=8.e-6)
parser.add_argument('--moving_delay', type=int, default=1000)
parser.add_argument('--warmup_delay', type=int, default=1000)
parser.add_argument('--pod_creation_time', type=int, default=100)
parser.add_argument('--data_sampler_cls', type=str, default='AlibabaDataSampler')
parser.add_argument('--mean_time_limit', type=float, default=2.e+8)
parser.add_argument('--train_data', type=str, default="AlibabaDataSampler")
parser.add_argument('--test_data', type=str, default="AlibabaDataSampler")
parser.add_argument('--render_mode', type=str, default="None")
parser.add_argument('--scheduler_name',type=str,default='HyperHeuristicScheduler')
parser.add_argument('--model_name', type=str, default="model_100_DNN")
parser.add_argument('--model_num_train', type=int,default=100)

# -- HyperHeuristic Scheduler -- #
parser.add_argument('--num_heuristics', type=int, default=2)
parser.add_argument('--list_heuristics', type=str, default=['MC','WSCPT'],nargs='+')
parser.add_argument('--input_feature', type=str, default=['num_queue',"glob" ],nargs='+')

# -- Resource allocation -- #
parser.add_argument('--resource_allocation', type=str, default='HyperHeuristic', choices=['Random', 'DNN', 'DRA', 'HyperHeuristic'])
parser.add_argument('--num_resource_heuristics', type=int, default= 2)
parser.add_argument('--list_resource_heuristics', type=str, default=['Maximum','DRA'],nargs='+')

# -- etc -- #
parser.add_argument('--splitting_rule', type=str, default="None")
parser.add_argument('--num_experiments', type=int, default=10)
#parser.add_argument('--plot_title', type = int, )
parser.add_argument('--input_file', type=str, default="ex_list.csv")
parser.add_argument('--result_folder', type=str, default="./test_results/0521/")
parser.add_argument('--output_file', type=str, default="result2.csv")

args = parser.parse_args()

def param_upate(par_name,par_value):
    for i in range(len(par_name)):
        if par_value[i] != "":
            vars(args)[par_name[i]]=par_value[i]