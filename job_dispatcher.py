import subprocess
import itertools
import multiprocessing

# Parameters
devices = ["0", "1", "2", "3", "4", "5"]  # Available GPUs
python_path = "/home/xxx/anaconda3/envs/xxx/bin/python"  # update this with the path to your conda
lora_rank = [1, 3, 5, 10, 20, 30, 40, 50, 60, 100, 150]
seed = [0, 42, 114514, 1919810, 20230822]

# Function to run tasks
def worker(device, queue):
    while True:
        # Get task from queue
        try:
            lora_rank, seed = queue.get(timeout=5)
        except multiprocessing.queues.Empty:
            break

        # The command to run
        cmd = f"{python_path} clm.py --soft_prompt_learning --learning_rate 1e-3 --n_tokens 20 --ZO --momentum --block_size 128 --zero_order_eps 1e-2 \
        --max_train_steps 10000 --per_device_train_batch_size 32 --zo_sample 4 --momentum_mu 0.9 --lora_more_rank \
            --lora_rank {lora_rank} --seed {seed} > results/lrank_{lora_rank}_seed_{seed}.txt"

        # Print the command
        print(f"Running command: {cmd} on GPU {device}")

        # Execute the command
        subprocess.run(cmd, shell=True, env={"CUDA_VISIBLE_DEVICES": device})

    print(f"Worker for device {device} finished")

# Create a queue for tasks
tasks = list(itertools.product(lora_rank, seed))
task_queue = multiprocessing.Queue()
for task in tasks:
    task_queue.put(task)

# Create a worker for each device
workers = []
for device in devices:
    worker_process = multiprocessing.Process(target=worker, args=(device, task_queue))
    workers.append(worker_process)
    worker_process.start()

# Wait for all workers to finish
for worker in workers:
    worker.join()
