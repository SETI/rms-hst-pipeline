from subprocess import CompletedProcess, run


def validate_bundle(deliverable_dir: str) -> None:
    completed_process: CompletedProcess = run(["./validate-pdart", deliverable_dir])
    completed_process.check_returncode()
    raise Exception("Succeeded but leaving a failure marker to prevent retries.")
