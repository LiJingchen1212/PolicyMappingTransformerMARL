

def make_env(scenario_name, benchmark=False, discrete_action=False):

    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as old_scenarios
    import envs.mpe_scenarios as new_scenarios

    # load scenario from script
    scenario = new_scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if hasattr(scenario, 'post_step'):
        post_step = scenario.post_step
    else:
        post_step = None
    if benchmark:        
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            info_callback=scenario.benchmark_data,
                            discrete_action=discrete_action)
    else:
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            discrete_action=discrete_action)
    return env
