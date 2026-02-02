import wandb

print("Trying wandb init for TEAM workspace...")
run = wandb.init(
    entity="SiT_AM",               # <-- NOTE: team entity
    project="sit-adjoint-matching",
    name="debug-team",
)
run.log({"ping": 1})
run.finish()
print("Done without exception.")

