import os
import subprocess
import time


def should_update_local(local_tag, remote_tag):
    if remote_tag[0] == local_tag[0]:
        return remote_tag != local_tag
    return False


def run_autoupdate(restart_script: str, process_pid: int):
    while True:
        local_tag = subprocess.getoutput("git describe --abbrev=0 --tags")
        os.system("git fetch")
        remote_tag = subprocess.getoutput(
            "git describe --tags `git rev-list --topo-order --tags HEAD --max-count=1`"
        )

        if should_update_local(local_tag, remote_tag):
            print("Local repo is not up-to-date. Updating...")
            reset_cmd = "git reset --hard " + remote_tag
            process = subprocess.Popen(reset_cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            if error:
                print("Error in updating:", error)
            else:
                print("Updated local repo to latest version: {}", format(remote_tag))

                print("Running the autoupdate steps...")
                # Trigger shell script. Make sure this file path starts from root
                subprocess.run(["kill", f"{process_pid}"], shell=True)
                subprocess.run(["wait", f"{process_pid}", "2>/dev/null"], shell=True)
                restart_process = subprocess.Popen([f"./{restart_script}"], shell=True)
                process_pid = restart_process.pid
                print("Finished running the autoupdate steps! Ready to go 😎")

        else:
            print("Repo is up-to-date.")

        time.sleep(10)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart_script", type=str)
    parser.add_argument("--process_pid", type=int)
    args = parser.parse_args()
    run_autoupdate(restart_script=args.restart_script, process_pid=args.process_pid)