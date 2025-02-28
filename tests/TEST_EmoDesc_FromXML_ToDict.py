import sys
import os

from ayaka_utils.Defs.pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.Defs.EmoDesc_FromXML_ToDict import EmoDesc_FromXML_ToDict

EmoDescFile = "./Configs/EmoList_Plutchik.xml"

test_en = EmoDesc_FromXML_ToDict(EmoDescFile, Lang="En")
test_ja = EmoDesc_FromXML_ToDict(EmoDescFile, Lang="Ja")

pprint("=" * 100)
pprint("English:")
pprint(test_en)
pprint("\n\n")
pprint("=" * 100)
pprint("Japanese:")
pprint(test_ja)