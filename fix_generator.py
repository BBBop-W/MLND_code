with open("src/data_processor/generator.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)

# Since doing a very complex structural refactor via string replace is risky, 
# I will output the whole block directly.

import re

# find block

