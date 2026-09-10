# PyHEOR examples

每个目录是一套可以独立运行的小研究：

- `markov_hiv/`：HIV 队列 Markov 模型
- `psm_oncology/`：肿瘤分区生存模型
- `microsim_sick_sicker/`：Sick-Sicker 个体模拟
- `multi_strategy_comparison/`：多策略成本效果比较

每个示例的入口都是 `example.py`，运行后生成的图保存在同目录的
`figures/`。适合用 Excel 独立复算的模型还会在 `workbooks/` 中生成
带公式的工作簿；它不是 Python 结果的静态截图。

在项目根目录安装开发版本后运行，例如：

```bash
pip install -e .
python examples/markov_hiv/example.py
```

HIV 示例包含随时间变化的状态成本，目前没有导出公式工作簿，因为现有
Excel 导出器还不能忠实表达这种成本。遇到不能复算的模型时，PyHEOR 会明确
报错，不会悄悄生成一个口径不同的文件。
