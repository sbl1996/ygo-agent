from ygoenv.registration import register

register(
  task_id="EDOPro-v0",
  import_path="ygoenv.edopro",
  spec_cls="EDOProEnvSpec",
  dm_cls="EDOProDMEnvPool",
  gym_cls="EDOProGymEnvPool",
  gymnasium_cls="EDOProGymnasiumEnvPool",
)
