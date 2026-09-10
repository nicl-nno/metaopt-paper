# Operational recovery, 2026-09-09

The originally session-bound campaign stopped at 384 searches. Detached processes
then reached 768 searches, but the network block 0 worker and the auditor each
raised `PermissionError: [WinError 5]` while replacing a JSON file. Later all
remaining calculation processes disappeared; the host had not rebooted. The exact
cause of their termination is not established. A successful launcher reset test
was insufficient to demonstrate persistence across the complete host lifecycle.

The preserved logs, partial training records and prior states are in
`results/recovery_20260909_v1` and `results/recovery_20260909_v2`. The completed
searches and selected model families are reused. Incomplete model families are
retrained with the original seeds, data and recipes; this adds training compute,
not new offline labels or extra online budget. Unsuccessful historical error
records are retained and must be interpreted together with completion markers.

`resilient.py` replaces only JSON persistence through a runtime adapter: a unique
temporary file is closed before replacement, and transient `PermissionError`
causes at most 30 bounded retries of the identical replacement. A persistent
failure still raises; existing output is not deleted, and file permissions are
not changed. Tests simulate transient and permanent errors and verify unchanged
JSON content and preservation of the previous result. No model, loss, descriptor,
oracle, seed, data split, validation rule, or evaluation budget changes.

All frozen original worker source hashes are still verified. The additional
adapter's SHA-256 is recorded in `operational_recovery.json`. Its scheduler adapter
routes each original worker launch through the same persistence wrapper. Audits
and analysis use that wrapper as well. This is an operational protocol deviation,
not a new scientific series or a method tuned on outcomes.

The workflow is now owned by a Windows Task Scheduler task named
`MetaOpt-SurrogateComparison-20260909`, using the current interactive user at
limited privilege, with no recurring trigger and no execution time limit. It
performs the remaining fixed campaign, complete audit, paired analysis and
publication in order. Up to three task restarts are configured after a failure;
each resumes the same saved allocation. No outcome-dependent stopping or
additional evaluation allocation is introduced. The task runs independently of
the chat tool and has been observed in Running state with live worker processes
and advancing training records. Closing the user login session or powering off
the host can still interrupt this interactive-user task.

Controller progress is in `controller_status.json`; worker progress and error
records remain in their block directories. `status.py` now reports process
liveness and heartbeat age rather than treating an old PID list as proof that
the calculation is running. Runtime logs of the Windows controller are in the
workspace's `surrogate_controller_20260909` directory; original worker logs remain
inside the campaign and are included in the final result snapshot.
