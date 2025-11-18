import os

root_dir = r"D:\dataset\murat_2018"
target_files = {"pyblinker_results.json", "pyblinker_results.pkl"}

for folder, subfolders, files in os.walk(root_dir):
	for filename in files:
		if filename in target_files:
			file_path = os.path.join(folder, filename)
			try:
				os.remove(file_path)
				print(f"Deleted: {file_path}")
			except Exception as e:
				print(f"Failed to delete {file_path}: {e}")

print("Done.")
