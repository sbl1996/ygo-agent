from ygoenv.registration import register

register(
  task_id="YGOPro-v0",
  import_path="ygoenv.ygopro0",
  spec_cls="YGOPro0EnvSpec",
  dm_cls="YGOPro0DMEnvPool",
  gym_cls="YGOPro0GymEnvPool",
  gymnasium_cls="YGOPro0GymnasiumEnvPool",
)
