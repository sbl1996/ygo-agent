from ygoenv.registration import register

register(
  task_id="YGOPro-v0",
  import_path="ygoenv.ygopro",
  spec_cls="YGOProEnvSpec",
  dm_cls="YGOProDMEnvPool",
  gym_cls="YGOProGymEnvPool",
  gymnasium_cls="YGOProGymnasiumEnvPool",
)
