from combined_yolo_training import create_combined_dataset_config, CombinedYOLOTrainer

trainer = CombinedYOLOTrainer()
valid_datasets = []
for dataset_id in trainer.dataset_manager.datasets.keys():
    dataset_config = trainer.dataset_manager.get_dataset_config(dataset_id)
    if dataset_config and trainer.dataset_manager.validate_dataset(dataset_config):
        valid_datasets.append(dataset_config)

print(f'Found {len(valid_datasets)} valid datasets')
for i, ds in enumerate(valid_datasets):
    print(f'{i+1}. {ds["name"]}')

# Create combined config
combined_config_path = create_combined_dataset_config(valid_datasets)
if combined_config_path:
    print(f'Combined dataset config created: {combined_config_path}')
else:
    print('Failed to create combined dataset configuration.')
