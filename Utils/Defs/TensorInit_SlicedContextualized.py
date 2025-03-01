# Python
from typing import List
import itertools

# Custom
from ayaka_utils.Classes.EmoTensorModels import EmoTensorFull_CTXD
from Utils.Defs.EmoDesc_FromXML_ToDict import EmoDesc_FromXML_ToDict

def TensorInit_SlicedContextualized(ConversationUsers: List[dict], EmoDescFile: str):
    """
    Initialize the sliced contextualized tensor
    """

    ConvUidsList = [user["id"] for user in ConversationUsers]
    
    # Sort the list of emoters by Uid order
    O4_Emoters = sorted(ConvUidsList, key=lambda x: int(x.split("=")[1]))

    Emoters_SpeakerOrder = sorted(ConversationUsers, key=lambda d: O4_Emoters.index(d['id']))
    PrefNames = [user["preferred_name"] for user in Emoters_SpeakerOrder]

    # Create the target list, it's a mesh of all the emoters
    O3_Target = [f"{a}-{b}" for a, b in itertools.permutations(O4_Emoters, 2)]

    # Pull in the EmoList_*.xml config file for emotion info
    try:
        EmoDescDict = EmoDesc_FromXML_ToDict(EmoDescFile, Lang="En")
    except Exception as e:
        raise Exception(f"Error parsing EmoDescFile: {e}")

    # Get the emotion families from the EmoList_*.xml config file
    O2_Emotions = [EmoDescDict[0]["EmoMeasures"][i]["Measure"] for i in range(len(EmoDescDict[0]["EmoMeasures"]))]

    # Get the attributes from the first emotion family in the EmoList_*.xml config file
    O1_Attributes = [EmoDescDict[0]["EmoMeasures"][i]["Measure"] for i in range(len(EmoDescDict[0]["EmoMeasures"]))]
    

    
    NewTensor = EmoTensorFull_CTXD(
        version="v0.0.1-EmoTensor Sliced Contextualized",
        order_1_attributes=O1_Attributes,
        order_2_emotions=O2_Emotions,
        order_3_target=O3_Target,
        order_4_emoters=O4_Emoters,
        order_5_transients=[],
        user_prefnames=PrefNames,
        transients=[]
    )
    return NewTensor
