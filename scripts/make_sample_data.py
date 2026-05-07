from ecg_ai_monitor.data.simulator import save_simulated_csv

if __name__ == "__main__":
    save_simulated_csv("data/sample_ecg.csv", duration_s=60, fs=250, scenario="mixed", seed=42)
    print("saved data/sample_ecg.csv")
