from trainer import Trainer

def build_env():

    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, AnyCondition
    from rlgym.rocket_league.action_parsers import RepeatAction
    from rlgym.rocket_league.rlviser.rlviser_renderer import RLViserRenderer

    from state_mutators.obs import Obs
    from state_mutators.action_parser import ActionParser

    termination_cond = GoalCondition()
    truncation_cond = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=30)
    )
  
    obs_builder = Obs()
    action_parser_base = ActionParser()
    action_parser = RepeatAction(action_parser_base, repeats=2)

  
    
    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_cond,
        truncation_cond=truncation_cond,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer()
    )
    return RLGymV2GymWrapper(rlgym_env)


if __name__ = "__main__":
    learner = Learner()
    learner.learn()
