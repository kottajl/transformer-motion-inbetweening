import subprocess
import sys

from scripts.seed_generator import get_n_seeds

SEEDS = get_n_seeds(1)

commands_to_run = [
    ...
]

def main():
    print("Starting commands queue...", file=sys.stderr)
    
    for idx, cmd in enumerate(commands_to_run, 1):
        cmd_str = " ".join(cmd)
        print(f"\n[{idx}/{len(commands_to_run)}] Executing: {cmd_str}", file=sys.stderr)
        print("-" * 50, file=sys.stderr)
        print(file=sys.stderr)
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"Error in execution!!!", file=sys.stderr)
            print("Stopping the queue...", file=sys.stderr)
            sys.exit(1)
            
        print(f"Command no {idx} has been finished succesfully", file=sys.stderr)
        print(file=sys.stderr)

    print("\n All commands has been finished succesfully! Good morning and have a tasty coffee.", file=sys.stderr)

if __name__ == "__main__":
    main()