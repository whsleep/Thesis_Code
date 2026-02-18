import os
import time
from sim import SIM_ENV

def run_batch_experiments():
    # --- 1. 实验配置 ---
    solvers_to_test = ["rda", "accdwa", "mpccbf", "mppi", "dwa", "teb"]   # 要测试的求解器列表
    # solvers_to_test = ["rda"]   # 要测试的求解器列表
    experiment_rounds = 100               # 每个求解器测试多少轮
    max_steps_per_episode = 1000        # 每轮最大步数
    render_mode = False                 # 是否显示画面
    base_log_dir = "experiment_results_5" # 根日志目录

    # --- 2. 开始循环测试 ---
    print(f"🚀 开始批量实验: 求解器={solvers_to_test}, 每组次数={experiment_rounds}")
    
    for solver_name in solvers_to_test:
        print(f"\n{'='*60}")
        print(f"🔵 当前测试求解器: {solver_name.upper()}")
        print(f"{'='*60}")

        # 为当前求解器创建专属日志目录
        current_log_path = os.path.join(base_log_dir, solver_name)
        if not os.path.exists(current_log_path):
            os.makedirs(current_log_path)

        # 初始化环境
        env = SIM_ENV(
            render=render_mode,
            save_ani=False,           
            solver_type=solver_name,
            log_path=current_log_path,
            timeout=100              
        )  

        # 循环执行实验 (使用普通 range，不再使用 tqdm)
        for round_idx in range(1, experiment_rounds + 1):
            
            # 定义每次实验的日志文件名 (例如: teb_exp_01)
            log_name = f"{solver_name}_exp_{round_idx:02d}"
            env.set_log_name(log_name)  # 将日志文件名传递给环境，以便记录器使用

            # 打印当前轮次信息
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] 🔹 正在执行: {solver_name} -> 第 {round_idx}/{experiment_rounds} 轮 ...")

            try:
                # 运行单次实验
                success = False
                for step in range(max_steps_per_episode + 2):
                    done = env.step()
                    
                    if done:
                        # 实验结束（碰撞或到达）
                        # print(f"    -> 结束于 step {step}") 
                        success = True
                        break
                
                if not success:
                    print(f"    ⚠️ 第 {round_idx} 轮超时 (达到最大步数)")
                
                env.logger.save_to_csv(f"{log_name}")
                env.reset()  # 重置环境和求解器，为下一轮做准备

            except Exception as e:
                print(f"\n❌ Error in {solver_name} round {round_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue # 继续下一轮

    print("\n✅ 所有实验已完成！")

if __name__ == "__main__":
    run_batch_experiments()