import os
import yaml

csvs = []
for r,d,f in os.walk("."):
    for name in f:
        if name == "afo_tune_device_0_full.csv":
            print(f"{r}/{name}")
            csvs.append(f"{r}/{name}")

combined_csv = ""
for csv in csvs:
    with open(csv, "r") as f:
        if combined_csv == "":
            combined_csv = f.read()
        else:
            for line in f:
                if not line.startswith("Validator"):
                    combined_csv += line
# Writeout combined files per device
for device in range(8):
    with open(f"afo_tune_device_{device}_full.csv", "w") as f:
        f.write(combined_csv)#
