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
    
    # Essential files to keep (absolute paths)
    essential_files = [
        os.path.join(tuberculosis_dir, "tb_dataset.py"),
        os.path.join(tuberculosis_dir, "tb_detection_comparison.ipynb"),
        os.path.join(tuberculosis_dir, "cleanup_files.py")  # Include this script
    ]
    
    # Copy all files to backup
    for filename in os.listdir(tuberculosis_dir):
        file_path = os.path.join(tuberculosis_dir, filename)
        
        # Skip directories that start with "backup_"
        if os.path.isdir(file_path) and filename.startswith("backup_"):
            print(f"Skipping existing backup directory: {filename}")
            continue
            
        # For directories, copy entire directory to backup
        if os.path.isdir(file_path):
            shutil.copytree(file_path, os.path.join(backup_dir, filename))
            print(f"Backed up directory: {filename}")
        # For files, copy to backup
        elif os.path.isfile(file_path):
            shutil.copy2(file_path, os.path.join(backup_dir, filename))
            print(f"Backed up file: {filename}")
    
    # Delete files that are not essential
    files_deleted = 0
    for filename in os.listdir(tuberculosis_dir):
        file_path = os.path.join(tuberculosis_dir, filename)
        
        # Skip essential files and backup directories
        if file_path in essential_files:
            print(f"Keeping essential file: {filename}")
            continue
            
        if os.path.isdir(file_path) and filename.startswith("backup_"):
            print(f"Keeping backup directory: {filename}")
            continue
        
        # Delete non-essential files
        if os.path.isfile(file_path):
            os.remove(file_path)
            files_deleted += 1
            print(f"Deleted file: {filename}")
        
        # Delete directories
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print(f"Deleted directory: {filename}")
    
    print(f"\nCleanup complete!")
    print(f"- {files_deleted} files deleted")
    print(f"- All files backed up to {backup_dir}")
    print(f"- Kept essential files: {', '.join(essential_files)}")
    print("\nYou now have a clean workspace with only the necessary files for TB detection.")

if __name__ == "__main__":
    # Confirmation prompt
    print("\n--- TUBERCULOSIS DIRECTORY CLEANUP ---")
    print("This script will:")
    print("1. Create a backup of all files")
    print("2. Delete ALL files and directories EXCEPT:")
    print("   - tb_dataset.py")
    print("   - tb_detection_comparison.ipynb")
    print("   - cleanup_files.py (this script)")
    print("   - backup_* directories")
    print("\nWARNING: This action cannot be undone except by restoring from the backup.")
    confirmation = input("\nDo you want to continue? (y/n): ")
    
    if confirmation.lower() == 'y':
        cleanup_files()
        print("\nCleanup completed successfully!")
        print("You can now safely commit your changes to your branch.")
    else:
        print("Operation cancelled.")
