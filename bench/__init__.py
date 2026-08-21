"""Benchmark for formwork.

Measures whether the library's three claims survive contact with a real model:
field ownership raises the first-try success rate, targeted repair costs fewer
tokens than regenerating, and declared repairs remove model calls entirely.

Every arm is scored by the same validator, so "valid" means one thing.
"""
