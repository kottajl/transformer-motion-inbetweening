import subprocess
import sys

from scripts.seed_generator import get_n_seeds

SEEDS = get_n_seeds(1)

commands_to_run = [
    ...
]

def main():
    print("Starting commands queue...")
    
    for idx, cmd in enumerate(commands_to_run, 1):
        cmd_str = " ".join(cmd)
        print(f"\n[{idx}/{len(commands_to_run)}] Executing: {cmd_str}")
        print("-" * 50)
        print()
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"Error in execution!!!")
            print("Stopping the queue...")
            sys.exit(1)
            
        print(f"Command no {idx} has been finished succesfully")
        print()

    print("\n All commands has been finished succesfully! Good morning and have a tasty coffee.")

if __name__ == "__main__":
    main()