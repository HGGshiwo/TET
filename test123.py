import asyncio
async def task(i):
    print(f"{i} start")
    await asyncio.sleep(10-i)
    print(f"{i} end")
    return i
    
async def main():
    tasks = [task(i) for i in range(5)]

    print(len(tasks))
    while tasks:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for each_task in done:
            each_task = await each_task
            print(f"{each_task} done")
            if each_task == 1:
                pending.add(task(9))
            if each_task == 2:
                pending.add(task(8))
        tasks = pending
        print(len(tasks))
        
asyncio.run(main())