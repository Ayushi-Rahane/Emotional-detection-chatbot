import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load saved reports produced earlier
baseline_path = os.path.join(base_dir, "../results/baseline_classification_report.xlsx")
transformer_path = os.path.join(base_dir, "../results/transformer_classification_report.xlsx")

baseline_df = pd.read_excel(baseline_path, index_col=0)
transformer_df = pd.read_excel(transformer_path, index_col=0)

# Export into one Excel file with two sheets
output_path = os.path.join(base_dir, "../results/model_comparison_reports.xlsx")

with pd.ExcelWriter(output_path) as writer:
    baseline_df.to_excel(writer, sheet_name="Baseline")
    transformer_df.to_excel(writer, sheet_name="Transformer")

print("Combined Excel report saved to:", output_path)
