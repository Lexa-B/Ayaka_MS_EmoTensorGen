import ast
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.Runnables.REmoScaleDesc import BuildReadableEmoScaleDesc
from ayaka_utils.Defs.pprint import pprint

EmoDescFile = "./Configs/EmoList_Plutchik.xml"

EmoScaleLabelsDefs = ast.literal_eval(open("./Configs/EmoScaleLabels.jsonc", "r").read())

output = BuildReadableEmoScaleDesc(EmoScaleLabelsDefs, "En")
pprint(output)