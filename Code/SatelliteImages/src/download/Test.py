import ee

# Initialize Earth Engine
ee.Initialize(project="wealth-satellite-forecasting")


# Get quota information - need to provide rootId
quota_info = ee.data.getAssetRootQuota(
    "wealth-satellite-forecasting"
)  # Replace with your actual username
print("Quota information:", quota_info)


# Get current task count
tasks = ee.data.getTaskList()
running_tasks = [task for task in tasks if task["state"] in ("RUNNING", "READY")]
print(f"Currently running/ready tasks: {len(running_tasks)}")
