import os

def apply_changes(decision_log_path:str):
    choices = []
    with open(decision_log_path, 'r') as f:
        decision = f.read().split('\n')
        decision.pop()
        for item in decision:
            choices.append(item.split(" > "))
    total = len(choices)
    for count, file in enumerate(choices):
        if file[1] == "delete":
            os.remove(path=file[0])
            print(f"Removed null file(number: {count+1})")
            continue
        os.rename(src=file[0], dst=file[1])
        print(f"Moved file {count+1}/{total}")
    
if __name__ == "__main__":
    apply_changes(input("Please give decision log path "))