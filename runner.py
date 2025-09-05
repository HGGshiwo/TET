from dataset.builder import build_dataset
from pathlib import Path
import decord
import asyncio
import multiprocessing as mp
mp.set_start_method('spawn', force=True) # for linux
from functools import partial
from utils import redirect_stdout

decord.bridge.set_bridge("torch")
from tqdm import tqdm
from utils import load_data, load_jsonl2dict, get_frame, LazyFrameLoader
import jsonlines


class Runner:
    def __init__(
        self,
        task,
        output_path=None,
        video_fps=None, # work when iter_frame is True
        iter_key="qid",
        iter_frame=False,
        filter = None,
        dataset="nextmc_test",
        dataset_config="./configs/dataset.yml",
        iter_callback=None,
        batch_size=1,
        max_workers=16,
        **kwargs
    ):
        for k, v in kwargs.items():
            assert not hasattr(self, k), f"{k} already exists"
            setattr(self, k, v)
        self.max_workers = max_workers
        self.dataset = build_dataset(dataset_config, dataset, is_training=False)
        # output_path = f"./outputs/{exp_name}/nextqa.jsonl"
        self.processed = {}
        if output_path is not None:
            if Path(output_path).exists():
                if iter_frame:
                    self.processed = load_jsonl2dict(output_path)
                else:
                    self.processed = load_data(output_path)
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).touch()

        self.iter_callback = iter_callback
        self.output_path = output_path
        self.iter_key = iter_key
        self.iter_frame = iter_frame
        if self.iter_frame is True:
            assert video_fps is not None, "video_fps must be set when iter_frame is True"
            self.video_fps = video_fps
        else:
            assert video_fps is None, "video_fps must be None when iter_frame is False"
        self.task = task
        self.tasks = []
        self.invalid = 0
        self.filter = filter
        self.batch_size = batch_size
        self.total = 0
        
    def create_submit(self, **kwargs):
        # return kwargs["excutor"].submit
        if self.batch_size == 1:
            def new_func(func, runner, **x):
                return lambda: func(runner, **x)
        else:
            def new_func(func, runner, data):
                return lambda: func(runner, data)
        return new_func

    def create_as_completed(self, **kwargs):
        # return concurrent.futures.as_completed
        def as_completed(tasks):
            for task in tasks:
                yield task()
        return as_completed

    def data_iter(self):
        data_iter = (
            self.dataset.get_video_info() if self.iter_key == "vid" else self.dataset
        )
        batch_data = []
        for data in data_iter:
            if data[self.iter_key] in self.processed:
                continue
            if self.filter is not None and self.filter(data) is False:
                continue
            self.total += 1
            if self.batch_size == 1:
                yield data
                continue
            batch_data.append(data)
            if len(batch_data) == self.batch_size:
                yield batch_data
                batch_data = []
        if len(batch_data) > 0:
            yield batch_data

    def frame_iter(self, use_tqdm=False):
        video_path = self.dataset.config.video_path
        data_iter = (
            self.dataset.get_video_info() if self.iter_key == "vid" else self.dataset
        )
        bar = None if not use_tqdm else tqdm(total=len(data_iter), desc="Generating frames")
        for data in data_iter:
            if data[self.iter_key] not in self.processed:
                _video_path = Path(video_path).joinpath(data["video_path"])
                batch_loader = LazyFrameLoader.create(_video_path, self.video_fps, self.batch_size)
                for d in batch_loader:
                    yield {"frame": d, **data.copy()}
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
            
            
    def handle_result(self, writer, result):
        if self.batch_size == 1:
            if result is not None:
                if writer is not None:
                    writer.write(result)
                if self.iter_callback is not None:
                    result = self.iter_callback(result)
            else:
                self.invalid += 1
        else:
            for res in result:
                if res is not None:
                    writer.write(res)
                else:
                    self.invalid += 1
                    
    def iter_loop(self, **kwargs):
        submit = self.create_submit(**kwargs)
        iter_func = self.frame_iter if self.iter_frame else self.data_iter
        for data in iter_func():
            self.tasks.append(submit(self.task, self, **data))


    def compelete_loop(self):
        as_completed = self.create_as_completed()
        self.bar = tqdm(as_completed(self.tasks), total=len(self.tasks))
        return self.bar

    def __call__(self):
        writer =  jsonlines.open(self.output_path, "a") if self.output_path is not None else None
        # with ProcessPoolExecutor(max_workers=1) as executor:
            # self.iter_loop(excutor=executor)
        self.iter_loop()
        for result in self.compelete_loop():
            with redirect_stdout():
                # result = result.result()
                self.handle_result(writer, result)
        print(f"Invalid: {self.invalid}/{len(self.tasks)}")
        if writer is not None:
            writer.close()


class AsyncRunner(Runner):
    def create_as_completed(self, **kwargs):
        return asyncio.as_completed

    def create_submit(self, **kwargs):
        sem = asyncio.Semaphore(self.max_workers)

        async def submit(task, runner, **kwargs):
            async with sem:
                result = await task(runner, **kwargs)
            return result

        return submit

    async def __call__(self):
        writer = jsonlines.open(self.output_path, "a") if self.output_path is not None else None
        sem = asyncio.Semaphore(200)
        self.iter_loop(sem=sem)
        for result in self.compelete_loop():
            with redirect_stdout():
                result = await result
            self.handle_result(writer, result)
        print(f"Invalid: {self.invalid}/{len(self.tasks)}")
        if writer is not None:
            writer.close()

import torch
class MultiGPURunner(Runner):
    def worker(self, gpu_id, model_cls):
        # 每个进程绑定自己的GPU
        torch.cuda.set_device(gpu_id)
        model = model_cls().to(f"cuda:{gpu_id}")
        while True:
            task_data = self.task_queue.get()
            if task_data is None:
                break
            if self.batch_size > 1:
                result = self.task(self, model=model, data=task_data)
            else:
                result = self.task(self, model=model, **task_data)
            self.result_queue.put(result)
        
    def __call__(self, model_class, gpu_ids=[]):
        manager = mp.Manager()
        self.task_queue = manager.Queue()
        self.result_queue = manager.Queue()

        # 1. 任务入队
        # iter_func = lambda: self.frame_iter(True) if self.iter_frame else self.data_iter
        iter_func = partial(self.frame_iter, True) if self.iter_frame else self.data_iter
        total = 0
        for data in iter_func():
            self.task_queue.put(data)
            total += 1
        # 2. 结束信号
        for _ in gpu_ids:
            self.task_queue.put(None)

        # 3. 启动进程
        processes = []
        for gpu_id in gpu_ids:
            p = mp.Process(target=self.worker, args=(gpu_id, model_class))
            p.start()
            processes.append(p)

        # 4. 收集结果
        bar = tqdm(total=total)
        writer = jsonlines.open(self.output_path, "a") if self.output_path is not None else None
        for _ in range(total):
            with redirect_stdout():
                result = self.result_queue.get()
            self.handle_result(writer, result)
            bar.update(1)
        bar.close()
        print(f"Invalid: {self.invalid}/{total}")
        if writer is not None:
            writer.close()

        for p in processes:
            p.join()