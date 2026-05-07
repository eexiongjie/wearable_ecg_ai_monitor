from __future__ import annotations


def compact_report(result: dict) -> str:
    gf = result.get("global_features", {})
    mean_hr = gf.get("mean_hr_bpm")
    sdnn = gf.get("sdnn_ms")
    rmssd = gf.get("rmssd_ms")
    return "\n".join(
        [
            "ECG AI Screening Report",
            "-----------------------",
            f"Duration: {result.get('duration_s'):.2f} s",
            f"Sampling rate: {result.get('fs')} Hz",
            f"Signal quality: {result.get('signal_quality')}",
            f"R peaks: {gf.get('r_peak_count')}",
            f"Mean HR: {mean_hr:.2f} bpm" if mean_hr is not None else "Mean HR: N/A",
            f"SDNN: {sdnn:.2f} ms" if sdnn is not None else "SDNN: N/A",
            f"RMSSD: {rmssd:.2f} ms" if rmssd is not None else "RMSSD: N/A",
            f"Summary: {result.get('summary_label')}",
            "Recommendations:",
            *[f"- {r}" for r in result.get("recommendations", [])],
        ]
    )
