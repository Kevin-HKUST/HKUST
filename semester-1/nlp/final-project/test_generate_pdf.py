import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

# 生成 PDF
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.9, '测试 PDF 用例\nSodium Content: 200 mg\n适合低钠饮食?', ha='center', va='center', fontsize=12)

# 表格
data = pd.DataFrame({
    '成分': ['Sodium (钠)', 'Calories (热量)', 'Fat (脂肪)'],
    '含量': ['200 mg', '100 kcal', '5 g']
})
table = ax.table(cellText=data.values, colLabels=data.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 保存 PDF
buf = BytesIO()
plt.savefig(buf, format='pdf', bbox_inches='tight')
buf.seek(0)
pdf_base64 = base64.b64encode(buf.read()).decode('utf-8')

# 保存到文件
with open('data/multimodal_files/test_pdf.pdf', 'wb') as f:
    f.write(base64.b64decode(pdf_base64))
print("PDF 生成成功: data/multimodal_files/test_pdf.pdf")
print("Base64 (前 100 chars):", pdf_base64[:100])