import os
import shutil
import datetime

def cleanup_files():
    """
    Clean up unnecessary files in the tuberculosis directory.
    This script:
    1. Creates a backup of all files in a timestamped folder
    2. Deletes unnecessary files
    3. Keeps only essential files for TB detection
    """
    # Current directory
    tuberculosis_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create backup directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(tuberculosis_dir, f"backup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"Creating backup in: {backup_dir}")
    
    # Essential files to keep
    essential_files = [
        "tb_dataset.py",
        "tb_detection_comparison.ipynb",
        "cleanup_files.py"  # Include this script
    ]
    
    # Copy all files to backup
    for filename in os.listdir(tuberculosis_dir):
        file_path = os.path.join(tuberculosis_dir, filename)
        
        # Skip directories and the backup dir itself
        if os.path.isdir(file_path) and filename != os.path.basename(backup_dir):
            # For directories, copy entire directory to backup
            shutil.copytree(file_path, os.path.join(backup_dir, filename))
            print(f"Backed up directory: {filename}")
        elif os.path.isfile(file_path):
            # For files, copy to backup
            shutil.copy2(file_path, os.path.join(backup_dir, filename))
            print(f"Backed up file: {filename}")
    
    # Delete files that are not essential
    files_deleted = 0
    for filename in os.listdir(tuberculosis_dir):
        file_path = os.path.join(tuberculosis_dir, filename)
        
        if os.path.isfile(file_path) and filename not in essential_files:
            os.remove(file_path)
            files_deleted += 1
            print(f"Deleted: {filename}")
        
        # Delete directories (except backup dir)
        if os.path.isdir(file_path) and filename != os.path.basename(backup_dir):
            shutil.rmtree(file_path)
            print(f"Deleted directory: {filename}")
    
    print(f"\nCleanup complete!")
    print(f"- {files_deleted} files deleted")
    print(f"- All files backed up to {backup_dir}")
    print(f"- Kept essential files: {', '.join(essential_files)}")
    print("\nYou now have a clean workspace with only the necessary files for TB detection.")

if __name__ == "__main__":
    # Confirmation prompt
    print("This script will delete unnecessary files in the tuberculosis directory.")
    print("A backup will be created before deletion.")
    confirmation = input("Do you want to continue? (y/n): ")
    
    if confirmation.lower() == 'y':
        cleanup_files()
    else:
        print("Operation cancelled.")
