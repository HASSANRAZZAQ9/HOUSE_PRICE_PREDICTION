"""Create a compact EDA report for the House Prices dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "train.csv")
report = []

report.append(f"Rows: {df.shape[0]}")
report.append(f"Columns: {df.shape[1]}")
report.append("\nTarget summary:")
report.append(df["SalePrice"].describe().to_string())

missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
report.append("\nMissing values:")
report.append(missing.to_string())

numeric = df.select_dtypes(exclude="object")
corr = numeric.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)
report.append("\nTop numeric correlations with SalePrice:")
report.append(corr.head(10).to_string())

out = ROOT / "reports" / "eda_report.txt"
out.write_text("\n".join(report), encoding="utf-8")
print(out)
