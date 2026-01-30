from sim import SIM_ENV

env = SIM_ENV(render=True, save_ani=False, solver_type="mpccbf", log_path="logs/mpccbf" , log_name="mpccbf_test", timeout=100)
# env = SIM_ENV(render=True, save_ani=False, solver_type="mppi", log_path="logs/mppi" , log_name="mppi_test", timeout=30)

for i in range(3000):
    if env.step():
        env.env.end(ending_time=i*0.1, suffix='.gif')
        break