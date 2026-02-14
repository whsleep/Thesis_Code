from sim import SIM_ENV

# env = SIM_ENV(render=True, save_ani=False, solver_type="mpccbf", log_path="logs/mpccbf" , log_name="mpccbf_test", timeout=100)
env = SIM_ENV(render=True, save_ani=False, solver_type="dwa", log_path="logs/dwa" , log_name="dwa_test", timeout=100)
# env = SIM_ENV(render=True, save_ani=False, solver_type="accdwa", log_path="logs/accdwa" , log_name="dwa_test", timeout=100)
# env = SIM_ENV(render=True, save_ani=False, solver_type="mppi", log_path="logs/mppi" , log_name="mppi_test", timeout=100)
# env = SIM_ENV(render=True, save_ani=False, solver_type="teb", log_path="logs/teb" , log_name="teb_test", timeout=100)
# env = SIM_ENV(render=True, save_ani=False, solver_type="rda", log_path="logs/rda", log_name="rda_test", timeout=100)

for i in range(3000):
    if env.step():
        env.env.end(ending_time=i*0.1, suffix='.gif')
        break