import time
import torch
import torch.multiprocessing as mp

from argparse import ArgumentParser
from cpu_adam_bench import CPUAdam_Benchmark

NUM_WARMUP=10

def worker_run_benchmark(rank, dtype, param_size, num_bench, shared_latency_variable, barrier, return_queue):  
    # print(f"Process {rank}| Benchmarking with dtype={dtype}, param_size={param_size}")
    # print(f"Process {rank}| Running benchmark: {NUM_WARMUP} warm-up steps, {num_bench} benchmark steps")
    benchmark_instance = CPUAdam_Benchmark(dtype, param_size)
    total_latency = 0.0
    
    for _ in range(NUM_WARMUP):
        benchmark_instance.step()

    for _ in range(num_bench):
        barrier.wait()  # make sure all process is ready to start
        start_time_ns = time.perf_counter_ns()
        benchmark_instance.step()
        end_time_ns = time.perf_counter_ns()
        
        latency_ns = end_time_ns - start_time_ns
        # print(f"Process {rank}| latency_ns: {latency_ns}")
        
        with shared_latency_variable.get_lock():
            if latency_ns > shared_latency_variable.value:
                shared_latency_variable.value = latency_ns

        barrier.wait()  # make sure all process finish checking maximum value
        total_latency += shared_latency_variable.value
        
        barrier.wait()  # make sure all process finish accumulation
        if rank == 0:
            shared_latency_variable.value = -1
            
    avg_Latency_per_step = total_latency / num_bench
    # print(f"Process {rank}| Average Latency per step: {avg_Latency_per_step:.6f} ns")
    return_queue.put((rank, avg_Latency_per_step))
  
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--nprocess",
        type=int,
        required=True
    )
    parser.add_argument(
        "--param_size",
        type=int,
        required=True
    )
    parser.add_argument(
        "--num_bench",
        type=int,
        required=True
    )
    args = parser.parse_args()
    
    nprocess = args.nprocess
    param_size = args.param_size
    num_bench = args.num_bench
    dtype = torch.float32
    
    aligned_nprocess_param_size = ((param_size + nprocess - 1) / nprocess) * nprocess
    partitioned_param_size = int(aligned_nprocess_param_size // nprocess)
    
    mp.set_start_method('spawn')
    shared_latency_variable = mp.Value('l', 0)
    barrier = mp.Barrier(nprocess)
    return_queue = mp.Queue()

    processes = []
    for rank in range(nprocess):
        p = mp.Process(target=worker_run_benchmark, args=(rank, dtype, partitioned_param_size, num_bench, shared_latency_variable, barrier, return_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    results = []
    for _ in range(nprocess):
        _, avg_Latency_per_step_local = return_queue.get()
        results.append(avg_Latency_per_step_local)
    assert len(set(results)) == 1, "The latency across different process is different!"
   
    avg_Latency_per_step = results[0]
    
    print(f"Benchmarking with benchmark steps={num_bench}, dtype={dtype}, param_size={param_size}, nprocess={nprocess}, partitioned_param_size={partitioned_param_size}")
    print(f"Average Latency per step: {avg_Latency_per_step/1000000:.6f} ms")